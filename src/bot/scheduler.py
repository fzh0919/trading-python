"""Timezone-aware weekday scheduler (Europe/London + US market DST/holiday adaptation) and post-close settlement."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.agents.llm_client import LLMBackend
from src.agents.pick_agent import generate_pick_radar
from src.agents.report_agent import (
    generate_intraday_alert,
    generate_morning_report,
    generate_us_sentinel,
)
from src.bot.telegram_client import TelegramClient
from src.config import AppConfig
from src.data_providers.analytics import _normalize_adjusted_ohlcv, resolve_market_benchmark
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import AccountSnapshot, PortfolioData, T212Client
from src.engine.calendar_utils import (
    add_trading_days,
    get_us_market_schedule_uk,
    has_market_closed_on_date,
    infer_market_from_symbol,
    is_eu_market_holiday,
    is_london_weekday,
    is_uk_bank_holiday,
)
from src.engine.screener import run_offline_universe_and_threshold_audit, run_universe_audition
from src.storage.db import AdvisorDatabase

if TYPE_CHECKING:
    from src.ops_notifier import OpsNotifier

logger = logging.getLogger(__name__)
LONDON_TZ = ZoneInfo("Europe/London")

MORNING_CATCHUP_END_MINUTES = 9 * 60 + 30  # 09:30 UK
DAILY_SETTLEMENT_START_MINUTES = 22 * 60  # 22:00 UK (after US/UK/EU close & Friday radar)
DAILY_SETTLEMENT_END_MINUTES = 23 * 60 + 30  # 23:30 UK catch-up cutoff
WEEKLY_AUDIT_START_MINUTES = 10 * 60  # 10:00 UK Saturday idle window
WEEKLY_AUDIT_END_MINUTES = 22 * 60  # 22:00 UK Saturday cutoff
INTRADAY_SENTINEL_START_MINUTES = 8 * 60  # 08:00 UK (LSE open)
INTRADAY_SENTINEL_END_MINUTES = 21 * 60  # 21:00 UK (US close)
WEEKLY_WALLET_RECONCILE_START_MINUTES = 12 * 60  # 12:00 UK Saturday ledger reminder window


def _parse_recommendation_date(rec: dict[str, Any]) -> date:
    """Extract the UK calendar date of a recommendation record."""
    created_uk = str(rec.get("created_at_uk") or "").strip()
    if len(created_uk) >= 10:
        try:
            return date.fromisoformat(created_uk[:10])
        except ValueError:
            pass
    created_ms = int(rec.get("created_at_ms") or 0)
    return datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).astimezone(LONDON_TZ).date()


def _resolve_benchmark_candidates(symbol: str, stored_benchmark: Any) -> list[str]:
    """Order stored benchmark first, then regional defaults from analytics."""
    candidates: list[str] = []
    stored = str(stored_benchmark or "").strip().upper()
    if stored:
        candidates.append(stored)
    candidates.extend(resolve_market_benchmark(symbol))
    return list(dict.fromkeys(c for c in candidates if c))


def _extract_closed_daily_ohlc(
    df: pd.DataFrame | None,
    market: str,
    now_utc: datetime,
) -> pd.DataFrame:
    """Return sorted daily High/Low/Close DataFrame keyed by calendar date for closed sessions."""
    empty = pd.DataFrame(columns=["High", "Low", "Close"])
    if df is None or df.empty or "Close" not in df.columns:
        return empty

    norm = _normalize_adjusted_ohlcv(df)
    if norm is None or norm.empty or "Close" not in norm.columns:
        return empty

    closes = pd.to_numeric(norm["Close"], errors="coerce").dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return empty

    highs = pd.to_numeric(norm["High"], errors="coerce") if "High" in norm.columns else closes
    lows = pd.to_numeric(norm["Low"], errors="coerce") if "Low" in norm.columns else closes

    idx = pd.to_datetime(closes.index)
    by_date: dict[date, tuple[float, float, float]] = {}
    for ts, close_val in zip(idx, closes.to_numpy(dtype=float)):
        if not math.isfinite(close_val) or close_val <= 0:
            continue
        bar_date = ts.date()
        if not has_market_closed_on_date(bar_date, market=market, now_utc=now_utc):
            continue
        raw_high = highs.loc[ts] if ts in highs.index else close_val
        raw_low = lows.loc[ts] if ts in lows.index else close_val
        if isinstance(raw_high, pd.Series):
            raw_high = raw_high.iloc[-1]
        if isinstance(raw_low, pd.Series):
            raw_low = raw_low.iloc[-1]
        h_val = float(raw_high) if pd.notna(raw_high) and math.isfinite(float(raw_high)) and float(raw_high) > 0 else float(close_val)
        l_val = float(raw_low) if pd.notna(raw_low) and math.isfinite(float(raw_low)) and float(raw_low) > 0 else float(close_val)
        by_date[bar_date] = (max(h_val, float(close_val)), min(l_val, float(close_val)), float(close_val))

    if not by_date:
        return empty
    ordered_dates = sorted(by_date.keys())
    return pd.DataFrame(
        [by_date[d] for d in ordered_dates],
        index=ordered_dates,
        columns=["High", "Low", "Close"],
        dtype=float,
    )


def _extract_closed_daily_closes(
    df: pd.DataFrame | None,
    market: str,
    now_utc: datetime,
) -> pd.Series:
    """Return sorted daily close prices keyed by calendar date for sessions already closed."""
    ohlc = _extract_closed_daily_ohlc(df, market=market, now_utc=now_utc)
    if ohlc.empty:
        return pd.Series(dtype=float)
    return ohlc["Close"]


def _align_lse_unit_100x(value: float, reference: float, symbol: str = "") -> float:
    """Align 100x GBp/GBP unit mismatch on London (.L) instruments against a reference price."""
    if symbol and not symbol.strip().upper().endswith(".L"):
        return value
    if value <= 0 or reference <= 0:
        return value
    ratio = reference / value
    if 20.0 <= ratio <= 200.0:
        return value * 100.0
    if 0.005 <= ratio <= 0.05:
        return value * 0.01
    return value


def _align_entry_price_gbp_guard(
    symbol: str,
    entry_price: float,
    stock_closes: pd.Series,
    rec_date: date,
    exit_price: float,
) -> float | None:
    """Align GBp/GBP 100x unit mismatch on .L stocks; return None if unresolved."""
    eff_entry = float(entry_price)
    if symbol.endswith(".L"):
        prior_dates = [d for d in stock_closes.index if d <= rec_date]
        if prior_dates and (rec_date - prior_dates[-1]).days <= 5:
            hist_entry = float(stock_closes.loc[prior_dates[-1]])
            eff_entry = _align_lse_unit_100x(eff_entry, hist_entry, symbol)
        exit_ratio = exit_price / eff_entry
        if 0.005 <= exit_ratio <= 0.05 or 20.0 <= exit_ratio <= 200.0:
            return None
    return eff_entry


def _compute_mfe_mae_pct(
    ohlc: pd.DataFrame,
    rec_date: date,
    exit_date: date,
    eff_entry: float,
) -> tuple[float | None, float | None]:
    """Compute Maximum Favorable Excursion (MFE%) and Maximum Adverse Excursion (MAE%) over (rec_date, exit_date]."""
    if ohlc.empty or eff_entry <= 0:
        return None, None
    dates = ohlc.index
    window = ohlc[(dates > rec_date) & (dates <= exit_date)]
    if window.empty:
        window = ohlc[dates == exit_date]
    if window.empty:
        return None, None
    max_high = float(window["High"].max())
    min_low = float(window["Low"].min())
    if not (math.isfinite(max_high) and math.isfinite(min_low)):
        return None, None
    mfe = round(((max_high - eff_entry) / eff_entry) * 100.0, 2)
    mae = round(((min_low - eff_entry) / eff_entry) * 100.0, 2)
    return mfe, mae


def _compute_benchmark_return_pct(
    bench_candidates: list[str],
    get_closes_fn: Any,
    rec_date: date,
    exit_date: date,
) -> tuple[str, float] | None:
    """Find the first valid regional benchmark and return (benchmark_symbol, return_pct)."""
    for b_sym in bench_candidates:
        b_market = infer_market_from_symbol(b_sym)
        b_closes = get_closes_fn(b_sym, b_market)
        if b_closes.empty or b_closes.index[0] > rec_date or b_closes.index[-1] < exit_date:
            continue
        b_entry_dates = [d for d in b_closes.index if d <= rec_date]
        b_exit_dates = [d for d in b_closes.index if d <= exit_date]
        if not b_entry_dates or not b_exit_dates:
            continue
        b_entry = float(b_closes.loc[b_entry_dates[-1]])
        b_exit = float(b_closes.loc[b_exit_dates[-1]])
        if b_entry > 0:
            return b_sym, ((b_exit - b_entry) / b_entry) * 100.0
    return None


def _clean_reflection_lesson(raw_text: str, max_chars: int = 200) -> str:
    """Strip markdown artifacts and keep the T+N reflection compact (<80 tokens)."""
    text = str(raw_text or "").replace("#", "").replace("**", "")
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    for sep in ("。", "！", "？", "."):
        pos = clipped.rfind(sep)
        if pos >= max_chars // 2:
            return clipped[: pos + 1].strip()
    return clipped.rstrip()


def _normalize_quote(text: str) -> str:
    """Normalize whitespace so verbatim quote checks tolerate spacing differences only."""
    return re.sub(r"\s+", "", str(text or ""))


def _quote_is_verbatim(quote: str, corpus: list[str]) -> bool:
    """True only when the extracted quote appears verbatim inside one raw user utterance."""
    needle = _normalize_quote(quote)
    if len(needle) < 4:
        return False
    return any(needle in _normalize_quote(item) for item in corpus)


def _to_float(value: Any) -> float | None:
    """Coerce to a finite positive float, else None (used for price / prev-close overlays)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) and num > 0 else None


def _intraday_move_pct(
    hist: pd.DataFrame | None,
    price: float | None,
    *,
    today_uk: date,
    prev_close: float | None = None,
    symbol: str = "",
) -> float | None:
    """Intraday move vs prior close; `prev_close` (Finnhub/T212) overrides the daily-bar baseline."""
    px = _to_float(price)
    if px is None:
        return None
    base = _to_float(prev_close)
    if base is None:
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna().astype(float)
        if closes.empty:
            return None
        try:
            last_bar_date = pd.Timestamp(closes.index[-1]).date()
        except (TypeError, ValueError):
            return None
        # Guard against multi-week stale caches masquerading as "intraday".
        if (today_uk - last_bar_date).days > 10:
            return None
        # A bar dated today is the running close -> the prior close is the bar before it.
        idx = -2 if (last_bar_date >= today_uk and len(closes) >= 2) else -1
        base = _to_float(closes.iloc[idx])
    if base is None:
        return None
    if symbol.strip().upper().endswith(".L"):
        px = _align_lse_unit_100x(px, base, symbol)
    return (px / base - 1.0) * 100.0


def _order_distance_pct(
    price: float | None,
    orders: list[Any],
    *,
    symbol: str = "",
) -> float | None:
    """Smallest absolute distance between the live price and any pending limit/stop trigger (%)."""
    px = _to_float(price)
    if px is None or not orders:
        return None
    distances: list[float] = []
    for order in orders:
        order_sym = symbol or str(getattr(order, "yahoo_symbol", "") or "")
        for field in ("limit_price", "stop_price"):
            trigger = _to_float(getattr(order, field, None))
            if trigger is None:
                continue
            if order_sym.strip().upper().endswith(".L"):
                trigger = _align_lse_unit_100x(trigger, px, order_sym)
            distances.append(abs(px - trigger) / px * 100.0)
    return min(distances) if distances else None


def _format_pending_orders(orders: list[Any]) -> str:
    """Compact plain-text rendering of a symbol's live pending orders."""
    lines: list[str] = []
    for order in orders[:3]:
        side = str(getattr(order, "side", "") or "").strip()
        quantity = getattr(order, "quantity", None)
        limit_price = getattr(order, "limit_price", None)
        currency = str(getattr(order, "currency", "") or "").strip()
        account = str(getattr(order, "account", "") or "").strip()
        lines.append(f"{account} {side} {quantity} @ {limit_price} {currency}".strip())
    return "; ".join(lines)


class TradingScheduler:
    """Deterministic minute-tick scheduler with per-job isolation and Telegram ops alerting."""

    def __init__(
        self,
        cfg: AppConfig,
        t212: T212Client,
        market_data: MarketDataService,
        llm: LLMBackend,
        db: AdvisorDatabase,
        telegram: TelegramClient,
        ops_notifier: OpsNotifier | None = None,
        execution_lock: asyncio.Lock | None = None,
        morning_retry_cooldown_s: float = 300.0,
    ) -> None:
        self.cfg = cfg
        self.t212 = t212
        self.market_data = market_data
        self.llm = llm
        self.db = db
        self.tg = telegram
        self.ops = ops_notifier
        self.execution_lock: asyncio.Lock = execution_lock if execution_lock is not None else asyncio.Lock()
        self.current_job_name: str | None = None
        self.morning_retry_cooldown_s = max(0.0, float(morning_retry_cooldown_s))
        self._fired_dates: dict[str, str] = {}
        self._morning_failures: dict[str, dict[str, Any]] = {}
        self._morning_missed_notified: set[str] = set()
        self.last_job_runs: dict[str, dict[str, str]] = {}
        self._last_intraday_scan_utc: datetime | None = None

    async def _is_job_locked(self, job_name: str, uk_date: str) -> bool:
        if self._fired_dates.get(job_name) == uk_date:
            return True
        if await self.db.has_scheduler_job_fired(job_name, uk_date):
            self._fired_dates[job_name] = uk_date
            return True
        return False

    async def _mark_job_locked(self, job_name: str, uk_date: str, uk_now_str: str) -> None:
        self._fired_dates[job_name] = uk_date
        await self.db.mark_scheduler_job_fired(job_name, uk_date, uk_now_str)

    async def _claim_job_run(self, job_name: str, uk_date: str, uk_now_str: str) -> bool:
        """Atomically check in-memory + SQLite persistence so restarts inside the trigger window never double-fire."""
        if await self._is_job_locked(job_name, uk_date):
            return False
        await self._mark_job_locked(job_name, uk_date, uk_now_str)
        return True

    async def _execute_job(
        self,
        *,
        job_name: str,
        chat_id: str,
        uk_now_str: str,
        coro: Awaitable[str],
        send_to_tg: bool = True,
        send_failure_to_tg: bool = True,
    ) -> bool:
        async with self.execution_lock:
            self.current_job_name = job_name
            try:
                msg = await coro
                if msg.lstrip().startswith("⚠️"):
                    self.last_job_runs[job_name] = {"status": "FAILED", "uk_time": uk_now_str, "detail": msg[:120]}
                    if self.ops is not None:
                        await self.ops.notify(
                            severity="CRITICAL",
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_job_failed:{job_name}",
                            message=f"定时任务 {job_name} 返回异常结果：{msg[:200]}",
                        )
                    if send_to_tg and send_failure_to_tg and chat_id:
                        await self.tg.send_message(chat_id, msg)
                    return False

                detail = msg[:120] if not send_to_tg else ""
                self.last_job_runs[job_name] = {"status": "OK", "uk_time": uk_now_str, "detail": detail}
                if send_to_tg and chat_id and msg.strip():
                    await self.tg.send_message(chat_id, msg)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduled job %s crashed", job_name)
                err_detail = f"{type(exc).__name__}: {exc}"
                self.last_job_runs[job_name] = {"status": "CRASHED", "uk_time": uk_now_str, "detail": err_detail[:120]}
                if self.ops is not None:
                    if job_name == "morning" and send_failure_to_tg and hasattr(self.ops, "notify_business_critical"):
                        await self.ops.notify_business_critical(
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_crash:{job_name}",
                            title="晨报定时任务异常崩溃",
                            message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                            bot_message="🚨 今日 /report 定时晨报生成异常中断，请发送 /report 重试或查看运维告警。",
                        )
                    else:
                        await self.ops.notify(
                            severity="CRITICAL",
                            source=f"scheduler:{job_name}",
                            alert_key=f"scheduler_crash:{job_name}",
                            message=f"定时任务 {job_name} 发生未捕获崩溃：{err_detail}",
                        )
                return False
            finally:
                if self.current_job_name == job_name:
                    self.current_job_name = None

    async def _generate_settlement_reflection(
        self,
        *,
        rec: dict[str, Any],
        rec_date: date,
        exit_date: date,
        holding_days: int,
        entry_price: float,
        exit_price: float,
        raw_return_pct: float,
        benchmark_symbol: str,
        bench_return_pct: float,
        alpha_pct: float,
    ) -> str:
        """Generate a concise 2-4 sentence (<80 tokens) T+N post-mortem lesson using Hydrogen LOW."""
        symbol = str(rec.get("symbol") or "").upper()
        account = str(rec.get("account") or "ISA")
        action = str(rec.get("action") or "WATCH")
        thesis = str(rec.get("thesis") or "").strip()[:220] or "未记录"

        system_instruction = (
            f"你是冷静客观的量化投研复盘教练。现在已知历史推荐在 T+{holding_days} 个交易日后的真实表现。\n"
            f"注意 T+{holding_days} 窗口可能短于中长线逻辑本身，不能仅凭短期波动全盘否定长线逻辑，但要直面短期择时或催化偏差。\n"
            "请用中文输出 2~4 句极简纯文本复盘教训（严格控制在 80 tokens / 120 汉字以内；"
            "严禁使用 #、**、列表符号或换行标题）：\n"
            f"1. 点明 T+{holding_days} 收益与相对 {benchmark_symbol} 的超额 Alpha 表现；\n"
            "2. 指出原推荐逻辑中哪部分被验证或证伪；\n"
            "3. 提炼 1 条下次同类选股或择时可直接复用的具体纪律。"
        )
        prompt = (
            f"标的: {symbol} ({account} | {action})\n"
            f"推荐日期: {rec_date.isoformat()} -> 结算日: {exit_date.isoformat()} (T+{holding_days})\n"
            f"入场价: {entry_price:.4g} -> 结算价: {exit_price:.4g}\n"
            f"原推荐逻辑: {thesis}\n"
            f"真实收益率: {raw_return_pct:+.2f}%\n"
            f"基准({benchmark_symbol})同期: {bench_return_pct:+.2f}%\n"
            f"超额 Alpha: {alpha_pct:+.2f}%"
        )
        raw_reply = await self.llm.generate(
            model=self.cfg.models.chat_model,
            prompt=prompt,
            system_instruction=system_instruction,
            thinking_level=self.cfg.models.chat_thinking_level,
            use_google_search=False,
            temperature=0.2,
            usage_label="settlement_reflection",
        )
        cleaned = _clean_reflection_lesson(raw_reply)
        if not cleaned or cleaned.lstrip().startswith("⚠️"):
            raise RuntimeError(cleaned or "empty reflection response")
        return cleaned

    def _has_audit_universe(self) -> bool:
        return bool(
            self.cfg.universe.audition_enabled
            or self.cfg.universe.isa_long_term
            or self.cfg.universe.invest_tactical
            or self.cfg.universe.etf_symbols
        )

    async def _generate_weekly_diagnosis(
        self,
        audit: dict[str, Any],
        *,
        uk_date: str,
    ) -> str:
        """One-hop DeepSeek diagnosis over raw audit numbers + raw T+N settlement stats (no summary-of-summary)."""
        if not ds_helper_available(self.llm, self.cfg):
            return ""
        try:
            stats = await self.db.get_recommendation_stats()
            lessons = await self.db.get_recent_lessons(limit=8)
        except Exception as exc:
            logger.warning("Weekly diagnosis could not load settlement stats: %s", exc)
            return ""

        bt = audit.get("threshold_backtest") or {}
        lines = [
            f"【本周体检 {uk_date}】池 {audit.get('universe_count', 0)} 只 | "
            f"重叠 {len(audit.get('overlap_candidates') or [])} | "
            f"僵尸 {len(audit.get('zombie_candidates') or [])}",
            f"【阈值回测原始数字】样本 {bt.get('sample_count', 0)} | {str(bt)[:600]}",
            f"【T+N 后验原始数字】总数 {stats.get('total_count', 0)} | 已结算 {stats.get('settled_count', 0)} | "
            f"原始胜率 {stats.get('win_rate_pct')} | Alpha 胜率 {stats.get('alpha_win_rate_pct')} | "
            f"T+10 Alpha {stats.get('avg_t10_alpha_pct')} | T+20 Alpha {stats.get('avg_t20_alpha_pct')} | "
            f"MFE {stats.get('avg_mfe_pct')} | MAE {stats.get('avg_mae_pct')}",
            "【近期已结算原话教训】",
        ]
        for row in lessons or []:
            lesson = str(row.get("reflection_lesson") or "").strip()
            if not lesson:
                continue
            lines.append(
                f"- {row.get('symbol')} {row.get('action')} 收益 {row.get('raw_return_pct')} "
                f"vs {row.get('benchmark_symbol')} Alpha {row.get('alpha_pct')}: {lesson[:120]}"
            )

        payload = await call_ds_json(
            self.cfg,
            self.llm,
            prompt="\n".join(lines),
            system_instruction=(
                "你是量化策略诊断官（只读）。根据原始回测与 T+5/10/20 后验数字，输出 3 条以内中文诊断："
                "①参数/阈值建议 ②胜率结构问题 ③下周关注。禁止编造未给出的数字，禁止 Markdown # 与 **，总长 <=450 字。"
                '仅输出严格 JSON：{"diagnosis": "..."}'
            ),
            usage_label="weekly_quant_diagnosis",
        )
        if not payload:
            return ""
        text = sanitize_plain_text(str(payload.get("diagnosis") or ""), 450)
        if len(text) < 20:
            return ""
        return text

    async def _run_weekly_quant_audit_impl(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Run 0-token offline universe audition + threshold audit and save silently to quant_health_audits."""
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")

        try:
            self.cfg.reload_pools_if_changed()
        except Exception as exc:
            logger.debug("Weekly quant audit config hot-reload failed: %s", exc)

        audition_res: dict[str, Any] | None = None
        if self.cfg.universe.audition_enabled and hasattr(self.db, "replace_active_universe_pool"):
            try:
                audition_res = await run_universe_audition(
                    self.cfg,
                    self.t212,
                    self.market_data,
                    self.db,
                    sync_online_indices=True,
                    sync_t212_metadata=True,
                )
            except Exception as exc:
                logger.warning("Weekly universe audition failed; falling back to existing pool: %s", exc)

        portfolio: PortfolioData | None = None
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Weekly quant audit portfolio fetch failed; falling back to snapshot: %s", exc)
        if portfolio is None:
            try:
                portfolio = await self.db.get_latest_snapshot()
            except Exception:
                portfolio = None
        if portfolio is None:
            portfolio = PortfolioData(
                fetched_at_uk=uk_now_str,
                fetched_at_ms=int(base.timestamp() * 1000),
                isa=AccountSnapshot(ok=False, account_label="ISA"),
                invest=AccountSnapshot(ok=False, account_label="Invest"),
                is_fallback=True,
                fallback_source="empty_fallback",
            )

        active_pool_rows: list[dict[str, Any]] = []
        if hasattr(self.db, "get_active_universe_pool"):
            try:
                active_pool_rows = await self.db.get_active_universe_pool()
            except Exception:
                active_pool_rows = []

        held_syms = portfolio.unique_yahoo_symbols(include_pending=False)
        if active_pool_rows:
            pool_syms = [str(r["symbol"]) for r in active_pool_rows] + list(self.cfg.universe.etf_symbols)
        else:
            pool_syms = (
                list(self.cfg.universe.isa_long_term)
                + list(self.cfg.universe.invest_tactical)
                + list(self.cfg.universe.etf_symbols)
            )
        all_syms = list(dict.fromkeys(s for s in (held_syms + pool_syms) if s))

        histories: dict[str, pd.DataFrame] = {}
        if all_syms:
            try:
                histories = await self.market_data.get_histories(all_syms, period="2y")
            except Exception as exc:
                logger.warning("Weekly quant audit histories fetch failed: %s", exc)

        audit_kwargs: dict[str, Any] = {
            "histories": histories,
            "portfolio": portfolio,
            "universe": self.cfg.universe,
            "strategy": self.cfg.strategy,
        }
        if active_pool_rows:
            audit_kwargs["active_pool_rows"] = active_pool_rows
        audit = run_offline_universe_and_threshold_audit(**audit_kwargs)
        if audition_res:
            audit["audition"] = audition_res
        diagnosis = await self._generate_weekly_diagnosis(audit, uk_date=uk_date)
        await self.db.save_quant_audit(
            audit_date=uk_date,
            summary_text=audit["summary_text"],
            payload=audit,
            created_at_uk=uk_now_str,
            ai_diagnosis=diagnosis,
        )
        return {**audit, "audit_date": uk_date, "ai_diagnosis": diagnosis}

    async def run_pool_audition(
        self,
        *,
        sync_online_indices: bool = True,
        sync_t212_metadata: bool = True,
        acquire_lock: bool = True,
    ) -> dict[str, Any]:
        """Refresh the curated 150-stock active universe pool (acquiring execution_lock unless already held)."""
        async def _run() -> dict[str, Any]:
            prev_job = self.current_job_name
            self.current_job_name = "refresh_pool"
            try:
                return await run_universe_audition(
                    self.cfg,
                    self.t212,
                    self.market_data,
                    self.db,
                    sync_online_indices=sync_online_indices,
                    sync_t212_metadata=sync_t212_metadata,
                )
            finally:
                if self.current_job_name == "refresh_pool":
                    self.current_job_name = prev_job

        if acquire_lock:
            async with self.execution_lock:
                return await _run()
        return await _run()

    async def run_weekly_quant_audit(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Acquire execution_lock and silently execute weekly offline universe & threshold audit."""
        async with self.execution_lock:
            self.current_job_name = "weekly_quant_audit"
            try:
                return await self._run_weekly_quant_audit_impl(now_utc=now_utc)
            finally:
                if self.current_job_name == "weekly_quant_audit":
                    self.current_job_name = None

    async def _run_weekly_quant_audit_job(self, now_utc: datetime) -> str:
        audit = await self._run_weekly_quant_audit_impl(now_utc=now_utc)
        bt = audit.get("threshold_backtest") or {}
        summary = (
            f"体检池 {audit.get('universe_count', 0)} 只 | "
            f"重叠 {len(audit.get('overlap_candidates') or [])} | "
            f"僵尸 {len(audit.get('zombie_candidates') or [])} | "
            f"回测样本 {bt.get('sample_count', 0)}"
        )
        diagnosis = str(audit.get("ai_diagnosis") or "").strip()
        if diagnosis:
            return f"📊 周六策略体检周报（{audit.get('audit_date', '')}）\n{diagnosis}\n\n{summary}"
        # Silent fallback: keep the DB audit as the source of truth and never spam a bare one-liner.
        logger.info("Weekly quant audit completed without AI diagnosis: %s", summary)
        return ""

    async def _build_weekly_wallet_reconcile_message(self, now_utc: datetime | None = None) -> str:
        """Compose a lightweight weekly Invest GBP+USD ledger reminder (0-token, no AI calls)."""
        dt_uk = (now_utc or datetime.now(timezone.utc)).astimezone(LONDON_TZ)
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            return f"⚠️ 钱包对账提醒生成失败：{exc}"
        invest = getattr(portfolio, "invest", None)
        if invest is None or not getattr(invest, "ok", False):
            return f"⚠️ Invest 账户读取失败，本周钱包对账提醒跳过：{getattr(invest, 'error', '未知错误')}"

        records: dict[str, dict[str, Any]] = {}
        if hasattr(self.db, "get_wallet_balances"):
            try:
                records = await self.db.get_wallet_balances("Invest")
            except Exception as exc:
                logger.warning("Weekly wallet reconcile: failed loading balances: %s", exc)
        usd = records.get("USD") or {}
        usd_total = float(usd.get("available_amount") or 0.0) + float(usd.get("reserved_amount") or 0.0)
        updated_uk = str(usd.get("updated_at_uk") or "")
        updated_ms = int(usd.get("updated_at_ms") or 0)
        age_days = (
            max(0, int((int(dt_uk.timestamp() * 1000) - updated_ms) / 86_400_000)) if updated_ms > 0 else None
        )
        rate: float | None = getattr(self.t212, "gbp_usd_rate", None)
        if rate is None and hasattr(self.market_data, "get_macro_snapshot"):
            try:
                macro = await self.market_data.get_macro_snapshot()
                candidate = getattr(macro, "gbp_usd", None)
                if candidate is not None and float(candidate) > 0:
                    rate = float(candidate)
            except Exception as exc:
                logger.warning("Weekly wallet reconcile: macro fetch failed: %s", exc)

        lines = ["🧾 每周 Invest 双钱包对账提醒"]
        usd_line = f"• 本地 USD 记录: ${usd_total:,.2f}"
        if updated_uk:
            usd_line += f"（{updated_uk} UK"
            usd_line += f"，{age_days} 天前）" if age_days is not None else "）"
        lines.append(usd_line)
        lines.append(f"• T212 实时总现金: £{float(invest.total_cash or 0.0):,.2f}")
        if rate:
            derived_gbp = max(0.0, float(invest.total_cash or 0.0) - usd_total / rate)
            lines.append(f"• 推算 GBP 钱包: ≈ £{derived_gbp:,.2f}（GBP/USD {rate:.4f}）")
        lines.append(
            "提示：若本周有换汇、美股买卖回笼或手动换币，请回复 /usd 金额 校准；"
            "也可 /wallet gbp 金额 单报，或 /wallet gbp x usd y 双报核对。"
        )
        return "\n".join(lines)

    async def _run_weekly_wallet_reconcile_job(self, now_utc: datetime) -> str:
        return await self._build_weekly_wallet_reconcile_message(now_utc=now_utc)

    async def _extract_preference_notes(self, now_utc: datetime | None = None) -> int:
        """Single-hop DeepSeek extraction of explicit user preferences from today's raw chat utterances."""
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        if not ds_helper_available(self.llm, self.cfg):
            return 0
        day_start = dt_uk.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            queries = await self.db.get_user_queries_since(int(day_start.timestamp() * 1000), limit=60)
        except Exception as exc:
            logger.warning("Preference extraction could not load chat history: %s", exc)
            return 0
        corpus = [q for q in queries if str(q).strip()]
        if not corpus:
            return 0

        joined = "\n---\n".join(f"[{idx + 1}] {q}" for idx, q in enumerate(corpus))
        payload = await call_ds_json(
            self.cfg,
            self.llm,
            prompt=f"【今日用户原始发言（逐条独立，禁止合并改写）】\n{joined[:6000]}",
            system_instruction=(
                "你是用户偏好抽取器（只读）。只抽取用户主动声明的明确硬偏好/资金计划/禁忌"
                "（例如「年底前不碰军工」「明年3月前不加仓科技」）。没有明确偏好就返回空数组。"
                "quote 必须是某一条发言的逐字片段（不要改写、不要合并），rule 是 <=60 字可执行规则，ttl_days 1~30。\n"
                '仅输出严格 JSON：{"notes": [{"quote": "...", "rule": "...", "ttl_days": 14}]}'
            ),
            usage_label="preference_extract",
        )
        if not payload:
            return 0
        raw_notes = payload.get("notes")
        if not isinstance(raw_notes, list):
            return 0

        max_notes = max(1, int(self.cfg.strategy.preference_max_notes_per_day))
        default_ttl = max(1, int(self.cfg.strategy.preference_ttl_days))
        validated: list[dict[str, Any]] = []
        for item in raw_notes[: max_notes * 2]:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            rule = sanitize_plain_text(str(item.get("rule") or ""), 60)
            if not quote or not rule or not _quote_is_verbatim(quote, corpus):
                if quote and not _quote_is_verbatim(quote, corpus):
                    logger.info("Dropping non-verbatim preference quote: %s", quote[:40])
                continue
            try:
                ttl = int(item.get("ttl_days") or default_ttl)
            except (TypeError, ValueError):
                ttl = default_ttl
            ttl = min(30, max(1, ttl))
            validated.append(
                {
                    "source_cycle_date": uk_date,
                    "quote_text": sanitize_plain_text(quote, 120),
                    "rule_summary": rule,
                    "expires_on_uk": (dt_uk.date() + timedelta(days=ttl)).isoformat(),
                }
            )
            if len(validated) >= max_notes:
                break
        if not validated:
            return 0
        saved = await self.db.upsert_preference_notes(validated)
        if saved:
            await self.db.prune_expired_preference_notes(uk_date)
        return saved

    async def _run_daily_settlement_impl(self, now_utc: datetime | None = None) -> dict[str, Any]:
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        pending = await self.db.get_pending_settlements(limit=50, only_with_entry_price=False)
        followups = (
            await self.db.get_settled_recommendations_for_followup(limit=50)
            if hasattr(self.db, "get_settled_recommendations_for_followup")
            else []
        )

        skipped = 0
        not_due = 0
        settled = 0
        failed = 0
        followup_updated = 0
        mature_items: list[tuple[dict[str, Any], date, date, str, int, float, list[str]]] = []
        followup_items: list[tuple[dict[str, Any], date, str, float, list[str], date | None, date | None]] = []

        for rec in pending:
            rec_id = int(rec["id"])
            symbol = str(rec.get("symbol") or "").strip().upper()
            entry_raw = rec.get("entry_price")
            bench_candidates = _resolve_benchmark_candidates(symbol, rec.get("benchmark_symbol"))

            try:
                entry_val = float(entry_raw) if entry_raw is not None else 0.0
            except (TypeError, ValueError):
                entry_val = 0.0

            if symbol in ("", "CASH") or not math.isfinite(entry_val) or entry_val <= 0:
                await self.db.settle_recommendation(
                    rec_id,
                    benchmark_symbol=bench_candidates[0] if bench_candidates else "SPY",
                    settled_at_uk=uk_now_str,
                    raw_return_pct=None,
                    alpha_pct=None,
                    outcome_status="SKIPPED",
                    reflection_lesson="",
                )
                skipped += 1
                continue

            rec_date = _parse_recommendation_date(rec)
            market = infer_market_from_symbol(symbol)
            holding_days = max(1, int(rec.get("horizon_days") or self.cfg.strategy.settlement_holding_days or 5))
            target_date = add_trading_days(rec_date, holding_days, market=market)
            if not has_market_closed_on_date(target_date, market=market, now_utc=base):
                not_due += 1
                continue

            mature_items.append((rec, rec_date, target_date, market, holding_days, entry_val, bench_candidates))

        for rec in followups:
            symbol = str(rec.get("symbol") or "").strip().upper()
            try:
                entry_val = float(rec.get("entry_price") or 0.0)
            except (TypeError, ValueError):
                entry_val = 0.0
            if symbol in ("", "CASH") or not math.isfinite(entry_val) or entry_val <= 0:
                continue
            rec_date = _parse_recommendation_date(rec)
            market = infer_market_from_symbol(symbol)
            bench_candidates = _resolve_benchmark_candidates(symbol, rec.get("benchmark_symbol"))

            t10_due_date: date | None = None
            if rec.get("t10_alpha_pct") is None:
                cand_t10 = add_trading_days(rec_date, 10, market=market)
                if has_market_closed_on_date(cand_t10, market=market, now_utc=base):
                    t10_due_date = cand_t10

            t20_due_date: date | None = None
            if rec.get("t20_alpha_pct") is None:
                cand_t20 = add_trading_days(rec_date, 20, market=market)
                if has_market_closed_on_date(cand_t20, market=market, now_utc=base):
                    t20_due_date = cand_t20

            if t10_due_date is not None or t20_due_date is not None:
                followup_items.append(
                    (rec, rec_date, market, entry_val, bench_candidates, t10_due_date, t20_due_date)
                )

        if mature_items or followup_items:
            fetch_symbols = list(
                dict.fromkeys(
                    [str(rec["symbol"]).strip().upper() for rec, *_ in mature_items]
                    + [b for *_, b_list in mature_items for b in b_list]
                    + [str(rec["symbol"]).strip().upper() for rec, *_ in followup_items]
                    + [b for _, _, _, _, b_list, _, _ in followup_items for b in b_list]
                )
            )
            try:
                histories = await self.market_data.get_histories(fetch_symbols, period="1y")
            except Exception as exc:
                logger.warning("Daily settlement history fetch failed: %s", exc)
                histories = {}
                failed += len(mature_items)
                mature_items = []
                followup_items = []

            ohlc_cache: dict[tuple[str, str], pd.DataFrame] = {}

            def _get_ohlc(sym: str, mkt: str) -> pd.DataFrame:
                key = (sym, mkt)
                if key not in ohlc_cache:
                    ohlc_cache[key] = _extract_closed_daily_ohlc(histories.get(sym), mkt, base)
                return ohlc_cache[key]

            def _get_closes(sym: str, mkt: str) -> pd.Series:
                ohlc = _get_ohlc(sym, mkt)
                return ohlc["Close"] if not ohlc.empty else pd.Series(dtype=float)

            for rec, rec_date, target_date, market, holding_days, entry_price, bench_candidates in mature_items:
                rec_id = int(rec["id"])
                symbol = str(rec["symbol"]).strip().upper()
                stock_ohlc = _get_ohlc(symbol, market)
                stock_closes = stock_ohlc["Close"] if not stock_ohlc.empty else pd.Series(dtype=float)
                exit_dates = [d for d in stock_closes.index if d >= target_date]
                if not exit_dates:
                    not_due += 1
                    continue
                exit_date = exit_dates[0]
                exit_price = float(stock_closes.loc[exit_date])

                eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, exit_price)
                if eff_entry is None:
                    logger.warning(
                        "Skipping settlement for %s (id=%s) due to unresolved GBp/GBP 100x mismatch",
                        symbol,
                        rec_id,
                    )
                    failed += 1
                    continue

                raw_return_pct = ((exit_price - eff_entry) / eff_entry) * 100.0
                bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, exit_date)
                if bench_res is None:
                    logger.warning("Benchmark history unavailable for settling %s (%s -> %s)", symbol, rec_date, exit_date)
                    failed += 1
                    continue
                chosen_bench, bench_ret_pct = bench_res
                alpha_pct = raw_return_pct - bench_ret_pct
                mfe_pct, mae_pct = _compute_mfe_mae_pct(stock_ohlc, rec_date, exit_date, eff_entry)

                try:
                    lesson = await self._generate_settlement_reflection(
                        rec=rec,
                        rec_date=rec_date,
                        exit_date=exit_date,
                        holding_days=holding_days,
                        entry_price=eff_entry,
                        exit_price=exit_price,
                        raw_return_pct=raw_return_pct,
                        benchmark_symbol=chosen_bench,
                        bench_return_pct=bench_ret_pct,
                        alpha_pct=alpha_pct,
                    )
                except Exception as exc:
                    logger.warning("Settlement reflection failed for %s (id=%s): %s", symbol, rec_id, exc)
                    failed += 1
                    continue

                saved = await self.db.settle_recommendation(
                    rec_id,
                    benchmark_symbol=chosen_bench,
                    settled_at_uk=uk_now_str,
                    raw_return_pct=round(raw_return_pct, 2),
                    alpha_pct=round(alpha_pct, 2),
                    outcome_status="SETTLED",
                    reflection_lesson=lesson,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                )
                if saved:
                    settled += 1

            for rec, rec_date, market, entry_price, bench_candidates, t10_target, t20_target in followup_items:
                rec_id = int(rec["id"])
                symbol = str(rec["symbol"]).strip().upper()
                stock_ohlc = _get_ohlc(symbol, market)
                if stock_ohlc.empty:
                    continue
                stock_closes = stock_ohlc["Close"]

                t10_ret: float | None = None
                t10_alpha: float | None = None
                t20_ret: float | None = None
                t20_alpha: float | None = None
                furthest_exit: date | None = None

                for horizon_label, target_d in (("t10", t10_target), ("t20", t20_target)):
                    if target_d is None:
                        continue
                    ex_dates = [d for d in stock_closes.index if d >= target_d]
                    if not ex_dates:
                        continue
                    ex_date = ex_dates[0]
                    ex_price = float(stock_closes.loc[ex_date])
                    eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, ex_price)
                    if eff_entry is None:
                        continue
                    raw_ret = ((ex_price - eff_entry) / eff_entry) * 100.0
                    bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, ex_date)
                    if bench_res is None:
                        continue
                    _, b_ret = bench_res
                    alpha_val = raw_ret - b_ret
                    if horizon_label == "t10":
                        t10_ret = round(raw_ret, 2)
                        t10_alpha = round(alpha_val, 2)
                    else:
                        t20_ret = round(raw_ret, 2)
                        t20_alpha = round(alpha_val, 2)
                    if furthest_exit is None or ex_date > furthest_exit:
                        furthest_exit = ex_date

                if t10_alpha is None and t20_alpha is None:
                    continue

                mfe_new: float | None = None
                mae_new: float | None = None
                if furthest_exit is not None:
                    ref_exit_price = float(stock_closes.loc[furthest_exit])
                    eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, ref_exit_price)
                    if eff_entry is not None:
                        mfe_new, mae_new = _compute_mfe_mae_pct(stock_ohlc, rec_date, furthest_exit, eff_entry)
                        prev_mfe = rec.get("mfe_pct")
                        prev_mae = rec.get("mae_pct")
                        if mfe_new is not None and prev_mfe is not None:
                            mfe_new = max(float(prev_mfe), mfe_new)
                        if mae_new is not None and prev_mae is not None:
                            mae_new = min(float(prev_mae), mae_new)

                updated = await self.db.update_recommendation_followup(
                    rec_id,
                    t10_return_pct=t10_ret,
                    t10_alpha_pct=t10_alpha,
                    t20_return_pct=t20_ret,
                    t20_alpha_pct=t20_alpha,
                    mfe_pct=mfe_new,
                    mae_pct=mae_new,
                )
                if updated:
                    followup_updated += 1

        weekly_audit_ran = False
        if self._has_audit_universe() and hasattr(self.db, "get_latest_quant_audit"):
            iso_year, iso_week, _ = dt_uk.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            latest_audit = await self.db.get_latest_quant_audit()
            audit_stale = latest_audit is None
            if latest_audit is not None:
                try:
                    audit_stale = (dt_uk.date() - date.fromisoformat(str(latest_audit["audit_date"])[:10])).days >= 7
                except ValueError:
                    audit_stale = True
            if audit_stale and await self._claim_job_run("weekly_quant_audit_catchup", week_key, uk_now_str):
                try:
                    await self._run_weekly_quant_audit_impl(now_utc=base)
                    weekly_audit_ran = True
                except Exception as exc:
                    logger.warning("Weekday catch-up weekly quant audit failed: %s", exc)

        pruned = await self.db.prune_old_records(retention_days=90, now_utc=base)
        preferences_saved = 0
        if self.cfg.strategy.user_preference_memory_enabled:
            try:
                preferences_saved = await self._extract_preference_notes(base)
            except Exception as exc:
                logger.warning("Preference extraction failed: %s", exc)
        return {
            "checked": len(pending),
            "settled": settled,
            "skipped": skipped,
            "not_due": not_due,
            "failed": failed,
            "followup_updated": followup_updated,
            "weekly_audit_ran": weekly_audit_ran,
            "pruned": pruned,
            "preferences_saved": preferences_saved,
        }

    async def run_daily_settlement(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Settle matured T+N recommendations with Alpha + Hydrogen LOW reflection and prune 90d logs."""
        async with self.execution_lock:
            self.current_job_name = "daily_settlement"
            try:
                return await self._run_daily_settlement_impl(now_utc=now_utc)
            finally:
                if self.current_job_name == "daily_settlement":
                    self.current_job_name = None

    async def _run_daily_settlement_job(self, now_utc: datetime) -> str:
        summary = await self._run_daily_settlement_impl(now_utc=now_utc)
        pruned_total = sum(summary["pruned"].values())
        if summary["settled"] == 0 and summary["failed"] > 0:
            return (
                f"⚠️ 盘后结算异常：待处理 {summary['checked']} 条中 {summary['failed']} 条失败，"
                f"清理 {pruned_total} 条过期记录"
            )
        return (
            f"结算 {summary['settled']} 条 / 追评 {summary.get('followup_updated', 0)} / "
            f"跳过 {summary['skipped']} / 未到期 {summary['not_due']} / 失败 {summary['failed']} / 清理 {pruned_total}"
        )

    async def _run_prewarm_impl(self, scope: str, now_utc: datetime | None = None) -> dict[str, Any]:
        """Silently warm 2y histories, macro snapshot, and daily 1x T212 instruments/dividends archive."""
        base = now_utc or datetime.now(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")

        try:
            self.cfg.reload_pools_if_changed()
        except Exception as exc:
            logger.debug("Prewarm config hot-reload failed: %s", exc)

        inst_synced = 0
        if hasattr(self.t212, "fetch_instruments_metadata"):
            try:
                if await self._claim_job_run("t212_metadata_sync", uk_date, uk_now_str):
                    inst_items = await self.t212.fetch_instruments_metadata()
                    if inst_items and hasattr(self.db, "upsert_t212_instruments"):
                        inst_synced = await self.db.upsert_t212_instruments(inst_items)
                    else:
                        inst_synced = len(inst_items or [])
            except Exception as exc:
                logger.warning("Prewarm T212 metadata sync failed (%s): %s", scope, exc)

        portfolio = None
        try:
            portfolio = await self.t212.get_portfolio(use_cache=False)
        except Exception as exc:
            logger.warning("Prewarm portfolio fetch failed (%s): %s", scope, exc)

        div_archived = 0
        if (
            portfolio is not None
            and not getattr(portfolio, "is_fallback", False)
            and hasattr(self.db, "upsert_dividends")
        ):
            live_divs = [
                d.model_dump()
                for acc in (portfolio.isa, portfolio.invest)
                if acc.ok and not getattr(acc, "is_fallback", False)
                for d in acc.recent_dividends
            ]
            if live_divs:
                try:
                    div_archived = await self.db.upsert_dividends(live_divs)
                except Exception as exc:
                    logger.warning("Prewarm dividend archive failed (%s): %s", scope, exc)

        held_syms = portfolio.unique_yahoo_symbols(include_pending=True) if portfolio is not None else []
        active_pool_rows: list[dict[str, Any]] = []
        if hasattr(self.db, "get_active_universe_pool"):
            try:
                active_pool_rows = await self.db.get_active_universe_pool()
            except Exception:
                active_pool_rows = []
        if active_pool_rows:
            pool_syms = (
                list(self.cfg.watchlist)
                + [str(r["symbol"]) for r in active_pool_rows]
                + list(self.cfg.universe.etf_symbols)
                + ["SPY", "QQQ", "ISF.L", "EXS1.DE"]
            )
        else:
            pool_syms = (
                list(self.cfg.watchlist)
                + list(self.cfg.universe.isa_long_term)
                + list(self.cfg.universe.invest_tactical)
                + list(self.cfg.universe.etf_symbols)
                + ["SPY", "QQQ", "ISF.L", "EXS1.DE"]
            )
        all_syms = list(dict.fromkeys(s for s in (held_syms + pool_syms) if s))

        hist_map: dict[str, pd.DataFrame] = {}
        if all_syms:
            try:
                hist_map = await self.market_data.get_histories(all_syms, period="2y")
            except Exception as exc:
                logger.warning("Prewarm histories fetch failed (%s): %s", scope, exc)

        macro_ok = False
        if hasattr(self.market_data, "get_macro_snapshot"):
            try:
                macro = await self.market_data.get_macro_snapshot()
                macro_ok = not getattr(macro, "is_stale", False)
            except Exception as exc:
                logger.warning("Prewarm macro snapshot failed (%s): %s", scope, exc)

        return {
            "scope": scope,
            "uk_date": uk_date,
            "symbols_warmed": sum(1 for df in hist_map.values() if df is not None and not df.empty),
            "instruments_synced": inst_synced,
            "dividends_archived": div_archived,
            "macro_ok": macro_ok,
        }

    async def run_prewarm(self, scope: str, now_utc: datetime | None = None) -> dict[str, Any]:
        """Acquire execution_lock and silently prewarm market caches & T212 archives."""
        job_key = f"prewarm_{scope}"
        async with self.execution_lock:
            self.current_job_name = job_key
            try:
                return await self._run_prewarm_impl(scope, now_utc=now_utc)
            finally:
                if self.current_job_name == job_key:
                    self.current_job_name = None

    async def _run_prewarm_job(self, scope: str, now_utc: datetime) -> str:
        summary = await self._run_prewarm_impl(scope, now_utc=now_utc)
        return (
            f"预热({scope}): 行情 {summary['symbols_warmed']} 只 | "
            f"元数据 {summary['instruments_synced']} 条 | 分红 {summary['dividends_archived']} 条"
        )

    def _intraday_scan_due(self, now_utc: datetime) -> bool:
        """Throttle the 0-token intraday scan to `intraday_poll_minutes`."""
        last = self._last_intraday_scan_utc
        if last is None:
            return True
        interval = timedelta(minutes=max(1, int(self.cfg.strategy.intraday_poll_minutes)))
        try:
            return (now_utc - last) >= interval
        except TypeError:
            return True

    async def _run_intraday_sentinel_job(
        self,
        now_utc: datetime,
        *,
        chat_id: str,
        uk_date: str,
    ) -> list[str]:
        """Run one intraday scan under the execution lock; returns symbols that were actually pushed."""
        async with self.execution_lock:
            prev_job = self.current_job_name
            self.current_job_name = "intraday_sentinel"
            self._last_intraday_scan_utc = now_utc
            try:
                alerts = await self._run_intraday_sentinel_impl(now_utc, uk_date=uk_date)
            except Exception:
                logger.exception("Intraday sentinel scan crashed; skipping this round")
                alerts = []
            finally:
                if self.current_job_name == "intraday_sentinel":
                    self.current_job_name = prev_job

        pushed: list[str] = []
        for sym, text in alerts:
            try:
                delivered = await self.tg.send_message(chat_id, text)
            except Exception as exc:  # noqa: BLE001 - delivery failure must not break the loop
                logger.warning("Intraday alert delivery failed for %s: %s", sym, exc)
                continue
            if delivered is False:
                logger.warning("Intraday alert for %s was not delivered; keeping today's dedupe slot free.", sym)
                continue
            # Mark only after a successful push so a quiet DS verdict never burns the daily dedupe slot.
            await self._mark_job_locked(
                f"intraday_alert:{sym}",
                uk_date,
                now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
            )
            pushed.append(sym)
        if pushed:
            self.last_job_runs["intraday_sentinel"] = {
                "status": "OK",
                "uk_time": now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
                "detail": ",".join(pushed)[:120],
            }
        return pushed

    async def _run_intraday_sentinel_impl(
        self,
        now_utc: datetime,
        *,
        uk_date: str,
    ) -> list[tuple[str, str]]:
        """0-token scan of open-market holdings/orders; only triggered symbols go to grounding + DS."""
        base = now_utc.astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Intraday sentinel portfolio fetch failed: %s", exc)
            return []
        if not portfolio.isa.ok and not portfolio.invest.ok:
            return []

        pending_by_symbol: dict[str, list[Any]] = {}
        for account in (portfolio.isa, portfolio.invest):
            for order in getattr(account, "pending_orders", None) or []:
                sym = str(getattr(order, "yahoo_symbol", "") or "").strip().upper()
                if sym:
                    pending_by_symbol.setdefault(sym, []).append(order)

        held_symbols = {
            str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            for pos in portfolio.all_positions()
            if str(getattr(pos, "yahoo_symbol", "") or "").strip()
        }
        candidates = sorted(held_symbols | set(pending_by_symbol))
        if not candidates:
            return []

        us_sched = get_us_market_schedule_uk(base)
        london_window = dt_uk.weekday() < 5 and time(8, 0) <= dt_uk.time() < time(16, 30)
        uk_holiday = is_uk_bank_holiday(base)
        eu_holiday = is_eu_market_holiday(base)
        market_open = {
            "US": us_sched.phase == "OPEN",
            "UK": london_window and not uk_holiday,
            "XETRA": london_window and not eu_holiday,
            "EURONEXT": london_window and not eu_holiday,
        }
        open_symbols = [s for s in candidates if market_open.get(infer_market_from_symbol(s), False)]
        if not open_symbols:
            return []

        try:
            histories, bundles = await self.market_data.get_bundles(
                open_symbols,
                include_news=False,
                include_earnings=False,
                include_info=False,
            )
        except Exception as exc:
            logger.warning("Intraday sentinel bundle fetch failed: %s", exc)
            return []

        # Finnhub fast US quote overlay: `include_info=False` bundles only carry the daily-close, which
        # can be 15+ minutes stale (h_cache TTL) or entirely missing right after the US open.
        quotes: dict[str, dict[str, Any]] = {}
        us_open_symbols = [s for s in open_symbols if infer_market_from_symbol(s) == "US"]
        if us_open_symbols:
            quote_results = await asyncio.gather(
                *(self.market_data.get_us_realtime_quote(s) for s in us_open_symbols),
                return_exceptions=True,
            )
            for sym, result in zip(us_open_symbols, quote_results):
                if isinstance(result, dict):
                    quotes[sym] = result
                elif isinstance(result, Exception):
                    logger.debug("Intraday realtime quote overlay failed for %s: %s", sym, result)

        position_by_symbol = {
            str(getattr(pos, "yahoo_symbol", "") or "").strip().upper(): pos
            for pos in portfolio.all_positions()
            if str(getattr(pos, "yahoo_symbol", "") or "").strip()
        }

        today_uk = dt_uk.date()
        triggered: list[tuple[float, str, float, float, float | None, str]] = []
        for sym in open_symbols:
            bundle = bundles.get(sym)
            quote = quotes.get(sym) or {}
            position = position_by_symbol.get(sym)
            # Live price precedence: Finnhub realtime -> T212 position mark -> daily-bar close.
            bundle_px = _to_float(getattr(bundle, "current_price", None))
            price = (
                _to_float(quote.get("current_price"))
                or _to_float(getattr(position, "cur_price", None))
                or bundle_px
            )
            if price is None:
                continue
            if sym.endswith(".L") and bundle_px is not None:
                price = _align_lse_unit_100x(price, bundle_px, sym)
            move_pct = _intraday_move_pct(
                histories.get(sym),
                price,
                today_uk=today_uk,
                prev_close=_to_float(quote.get("prev_close")),
                symbol=sym,
            )
            order_dist = _order_distance_pct(price, pending_by_symbol.get(sym) or [], symbol=sym)
            hit_move = move_pct is not None and abs(move_pct) >= self.cfg.strategy.intraday_move_alert_pct
            hit_order = order_dist is not None and order_dist <= self.cfg.strategy.intraday_order_near_pct
            if hit_move or hit_order:
                currency = str(
                    getattr(bundle, "currency", "") or getattr(position, "inst_currency", "") or ""
                )
                triggered.append(
                    (abs(move_pct or 0.0), sym, price, float(move_pct or 0.0), order_dist, currency)
                )

        triggered.sort(key=lambda item: item[0], reverse=True)
        max_alerts = 2
        alerts: list[tuple[str, str]] = []
        for _, sym, price, move_pct, order_dist, currency in triggered:
            if len(alerts) >= max_alerts:
                logger.info("Intraday sentinel capped at %d alerts for this round", max_alerts)
                break
            if await self._is_job_locked(f"intraday_alert:{sym}", uk_date):
                continue
            try:
                text = await generate_intraday_alert(
                    self.cfg,
                    self.market_data,
                    self.llm,
                    self.db,
                    symbol=sym,
                    current_price=price,
                    move_pct=move_pct,
                    order_dist_pct=order_dist,
                    pending_order_text=_format_pending_orders(pending_by_symbol.get(sym) or []),
                    currency=currency,
                    ops_notifier=self.ops,
                )
            except Exception as exc:  # noqa: BLE001 - one symbol must never break the scan
                logger.warning("Intraday alert generation failed for %s: %s", sym, exc)
                text = None
            if text:
                alerts.append((sym, text))
        return alerts

    async def check_and_run_due_jobs(self, now_utc: datetime | None = None) -> list[str]:
        """Evaluate prewarm, morning catch-up, premarket sentinel, weekly radar, Saturday quant audit, and post-close settlement."""
        base = now_utc or datetime.now(timezone.utc)
        chat_id = self.cfg.env.telegram_chat_id.strip()
        if not chat_id:
            return []

        dt_uk = base.astimezone(LONDON_TZ)
        uk_date = dt_uk.date().isoformat()
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        uk_min_of_day = dt_uk.hour * 60 + dt_uk.minute
        fired: list[str] = []

        # Saturday idle window (10:00 ~ 22:00 UK): silent 0-token offline universe & threshold health audit
        if dt_uk.weekday() == 5:
            if WEEKLY_AUDIT_START_MINUTES <= uk_min_of_day < WEEKLY_AUDIT_END_MINUTES and self._has_audit_universe():
                iso_year, iso_week, _ = dt_uk.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                if await self._claim_job_run("weekly_quant_audit", week_key, uk_now_str):
                    fired.append("weekly_quant_audit")
                    logger.info("Running Saturday silent weekly quant health audit (%s)", week_key)
                    await self._execute_job(
                        job_name="weekly_quant_audit",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_weekly_quant_audit_job(base),
                        send_to_tg=True,
                    )

            # Saturday ledger reminder window (12:00 ~ 22:00 UK): weekly Idempotent GBP+USD reconcile nudge
            if WEEKLY_WALLET_RECONCILE_START_MINUTES <= uk_min_of_day < WEEKLY_AUDIT_END_MINUTES:
                iso_year, iso_week, _ = dt_uk.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                if await self._claim_job_run("weekly_wallet_reconcile", week_key, uk_now_str):
                    fired.append("weekly_wallet_reconcile")
                    logger.info("Sending weekly wallet reconcile reminder (%s)", week_key)
                    await self._execute_job(
                        job_name="weekly_wallet_reconcile",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_weekly_wallet_reconcile_job(base),
                    )
            return fired

        if not is_london_weekday(base):
            return []

        # 0. Silent Morning Prewarm (-10m before morning_report, e.g. 07:20 ~ 07:29 UK)
        morning_target = self.cfg.schedule.morning_report_hour * 60 + self.cfg.schedule.morning_report_minute
        morning_prewarm_target = max(0, morning_target - 10)
        if morning_prewarm_target <= uk_min_of_day < morning_target:
            if await self._claim_job_run("prewarm_morning", uk_date, uk_now_str):
                fired.append("prewarm_morning")
                logger.info("Running silent morning prewarm (%s)", uk_date)
                await self._execute_job(
                    job_name="prewarm_morning",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_prewarm_job("morning", base),
                    send_to_tg=False,
                )

        # 1. Morning Report with N95 Catch-up & Retry (07:30 ~ 09:30 UK; lock scheduler_runs only on success)
        morning_catchup_end = max(morning_target + 5, MORNING_CATCHUP_END_MINUTES)
        if morning_target <= uk_min_of_day < morning_catchup_end:
            if not await self._is_job_locked("morning", uk_date):
                retry_info = self._morning_failures.get(uk_date)
                if retry_info is None or base >= retry_info["next_retry_utc"]:
                    fired.append("morning")
                    logger.info("Running scheduled morning report (%s)", uk_date)
                    ok = await self._execute_job(
                        job_name="morning",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_morning_report(
                            self.cfg, self.t212, self.market_data, self.llm, self.db, ops_notifier=self.ops
                        ),
                        send_failure_to_tg=False,
                    )
                    if ok:
                        await self._mark_job_locked("morning", uk_date, uk_now_str)
                        self._morning_failures.pop(uk_date, None)
                    else:
                        attempts = int(retry_info["attempts"]) + 1 if retry_info else 1
                        self._morning_failures[uk_date] = {
                            "attempts": attempts,
                            "next_retry_utc": base + timedelta(seconds=self.morning_retry_cooldown_s),
                        }
        elif uk_min_of_day >= morning_catchup_end:
            if (
                uk_date in self._morning_failures
                and uk_date not in self._morning_missed_notified
                and not await self._is_job_locked("morning", uk_date)
            ):
                self._morning_missed_notified.add(uk_date)
                if self.ops is not None and hasattr(self.ops, "notify_business_critical"):
                    await self.ops.notify_business_critical(
                        source="scheduler:morning",
                        alert_key=f"scheduler_morning_catchup_exhausted:{uk_date}",
                        title="晨报自动补发窗口已结束",
                        message=f"{uk_date} 早报在 09:30 UK 前多次补发仍未成功，请尽快检查网络或接口状态。",
                        bot_message="🚨 今日 /report 早报自动补发窗口（09:30 UK）已结束且仍未成功，请手动发送 /report。",
                    )
                else:
                    await self.tg.send_message(
                        chat_id,
                        "🚨 今日 /report 早报自动补发窗口（09:30 UK）已结束且仍未成功，请手动发送 /report。",
                    )

        # 2. US Pre-Market Sentinel & -10m Silent Prewarm (Mon-Fri, dynamic DST-adjusted minutes_to_open)
        us_sched = get_us_market_schedule_uk(base)
        lead = self.cfg.schedule.premarket_lead_minutes
        if us_sched.is_us_trading_day and not us_sched.is_us_holiday and lead < us_sched.minutes_to_open <= lead + 10:
            if await self._claim_job_run("prewarm_premarket", uk_date, uk_now_str):
                fired.append("prewarm_premarket")
                logger.info("Running silent premarket prewarm (%s)", uk_date)
                await self._execute_job(
                    job_name="prewarm_premarket",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_prewarm_job("premarket", base),
                    send_to_tg=False,
                )

        if us_sched.minutes_to_open <= lead and us_sched.minutes_to_open > lead - 5:
            if await self._claim_job_run("premarket", uk_date, uk_now_str):
                fired.append("premarket")
                if us_sched.is_us_holiday:
                    logger.info("US Market Holiday (%s): sending quiet notice.", us_sched.ny_date)
                    self.last_job_runs["premarket"] = {
                        "status": "HOLIDAY_MUTED",
                        "uk_time": uk_now_str,
                        "detail": us_sched.ny_date,
                    }
                    await self.tg.send_message(
                        chat_id,
                        "🔔 今日为美国股市法定休市日（Market Holiday），美股盘前哨兵自动静音，祝你今天轻松愉快！",
                    )
                elif us_sched.is_us_trading_day:
                    logger.info("Running scheduled US pre-market sentinel (%s)", uk_date)
                    await self._execute_job(
                        job_name="premarket",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_us_sentinel(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.llm,
                            self.db,
                            is_scheduled_premarket=True,
                            ops_notifier=self.ops,
                        ),
                    )

        # 3. Friday Post-Close Weekly Opportunity Radar & -10m Silent Prewarm (respects dynamic early-close)
        if self.cfg.schedule.enable_weekly_radar and dt_uk.weekday() == 4 and us_sched.is_us_trading_day:
            after = self.cfg.schedule.weekly_radar_after_close_minutes
            mins_after_close = -us_sched.minutes_to_close
            prewarm_after = max(0, after - 10)
            if prewarm_after <= mins_after_close < after:
                if await self._claim_job_run("prewarm_weekly_radar", uk_date, uk_now_str):
                    fired.append("prewarm_weekly_radar")
                    logger.info("Running silent weekly radar prewarm (%s)", uk_date)
                    await self._execute_job(
                        job_name="prewarm_weekly_radar",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=self._run_prewarm_job("weekly_radar", base),
                        send_to_tg=False,
                    )
            if after <= mins_after_close < after + 5:
                if await self._claim_job_run("weekly_radar", uk_date, uk_now_str):
                    fired.append("weekly_radar")
                    logger.info("Running scheduled Friday weekly opportunity radar (%s)", uk_date)
                    await self._execute_job(
                        job_name="weekly_radar",
                        chat_id=chat_id,
                        uk_now_str=uk_now_str,
                        coro=generate_pick_radar(
                            self.cfg,
                            self.t212,
                            self.market_data,
                            self.llm,
                            self.db,
                            focus_query="",
                            is_weekly_auto=True,
                            ops_notifier=self.ops,
                        ),
                    )

        # 4. Daily Post-Close T+N Recommendation Settlement & 90-Day Pruning (22:00 ~ 23:30 UK)
        if DAILY_SETTLEMENT_START_MINUTES <= uk_min_of_day < DAILY_SETTLEMENT_END_MINUTES:
            if await self._claim_job_run("daily_settlement", uk_date, uk_now_str):
                fired.append("daily_settlement")
                logger.info("Running scheduled daily T+N settlement & DB pruning (%s)", uk_date)
                await self._execute_job(
                    job_name="daily_settlement",
                    chat_id=chat_id,
                    uk_now_str=uk_now_str,
                    coro=self._run_daily_settlement_job(base),
                    send_to_tg=False,
                )

        # 5. Intraday event-driven sentinel (08:00~21:00 UK; 0-token scan, never queues behind user commands)
        if (
            self.cfg.strategy.intraday_sentinel_enabled
            and INTRADAY_SENTINEL_START_MINUTES <= uk_min_of_day < INTRADAY_SENTINEL_END_MINUTES
            and not self.execution_lock.locked()
            and self._intraday_scan_due(base)
        ):
            pushed = await self._run_intraday_sentinel_job(base, chat_id=chat_id, uk_date=uk_date)
            if pushed:
                fired.append("intraday_sentinel")

        return fired

    async def run_forever(self, poll_interval_s: float = 30.0) -> None:
        logger.info("TradingScheduler started (Europe/London timezone aware).")
        while True:
            try:
                await self.check_and_run_due_jobs()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduled job loop error")
                if self.ops is not None:
                    await self.ops.notify(
                        severity="CRITICAL",
                        source="scheduler:loop",
                        alert_key="scheduler_loop_error",
                        message=f"调度主循环异常：{type(exc).__name__}: {exc}",
                    )
            await asyncio.sleep(poll_interval_s)
