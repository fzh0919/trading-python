"""Daily post-close T+5/10/20 settlement, NAV snapshot, reflection, and preference-memory mixin."""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.bot.scheduler.fill_receipts import _normalize_account_label
from src.bot.scheduler.postmarket_mixin import PostmarketReviewMixin
from src.bot.scheduler.settlement_math import (
    LONDON_TZ,
    NAV_SNAPSHOT_START_MINUTES,
    _align_entry_price_gbp_guard,
    _clean_reflection_lesson,
    _compute_benchmark_return_pct,
    _compute_mfe_mae_pct,
    _direction_adjusted_alpha_pct,
    _direction_adjusted_return_pct,
    _entry_fill_status,
    _extract_closed_daily_ohlc,
    _match_recommendation_fill,
    _parse_recommendation_date,
    _quote_is_verbatim,
    _resolve_benchmark_candidates,
    _trade_direction,
)
from src.data_providers.analytics import (
    build_nav_row,
    classify_process_quadrant,
    reconcile_recent_nav_flows,
)
from src.engine.calendar_utils import (
    add_trading_days,
    has_market_closed_on_date,
    infer_market_from_symbol,
)

logger = logging.getLogger("src.bot.scheduler")


class SettlementJobMixin(PostmarketReviewMixin):
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
        direction: int = 1,
        mfe_pct: float | None = None,
        mae_pct: float | None = None,
        actual_filled: int | None = None,
        actual_fill_price: float | None = None,
        execution_slippage_pct: float | None = None,
        regime: str = "",
        process_quadrant: str = "",
        holding_catalyst: str = "",
    ) -> str:
        """Generate a concise 2-4 sentence (<80 tokens) T+N post-mortem lesson using Hydrogen LOW."""
        symbol = str(rec.get("symbol") or "").upper()
        account = str(rec.get("account") or "ISA")
        action = str(rec.get("action") or "WATCH")
        thesis = str(rec.get("thesis") or "").strip()[:220] or "未记录"
        direction_label = "多头买入（价格涨=正确）" if direction >= 0 else "减仓/止损（价格跌=正确，避损口径）"

        extra_lines: list[str] = []
        if str(regime or "").strip():
            extra_lines.append(f"市场状态: {str(regime).strip().upper()}")
        if mfe_pct is not None or mae_pct is not None:
            mfe_txt = f"{float(mfe_pct):+.2f}%" if mfe_pct is not None else "N/A"
            mae_txt = f"{float(mae_pct):+.2f}%" if mae_pct is not None else "N/A"
            extra_lines.append(f"持有期 MFE / MAE: {mfe_txt} / {mae_txt}")
        if actual_filled is not None or execution_slippage_pct is not None:
            filled_txt = "是" if actual_filled == 1 else ("否" if actual_filled == 0 else "未核对")
            fill_price_txt = f"{float(actual_fill_price):.4g}" if actual_fill_price is not None else "N/A"
            slip_txt = (
                f"{float(execution_slippage_pct):+.2f}%"
                if execution_slippage_pct is not None
                else "N/A"
            )
            extra_lines.append(f"实盘成交与滑点: 成交 {filled_txt} | 成交价 {fill_price_txt} | 滑点 {slip_txt}")
        quadrant_txt = str(process_quadrant or "").strip().upper()
        if quadrant_txt:
            extra_lines.append(f"2×2 过程象限: {quadrant_txt}")
        catalyst_txt = str(holding_catalyst or "").strip()
        if catalyst_txt:
            extra_lines.append(f"持有期客观催化: {catalyst_txt[:160]}")

        system_instruction = (
            f"你是冷静客观的量化投研复盘教练。现在已知历史推荐在 T+{holding_days} 个交易日后的真实表现。\n"
            f"注意 T+{holding_days} 窗口可能短于中长线逻辑本身，不能仅凭短期波动全盘否定长线逻辑，但要直面短期择时或催化偏差。\n"
            "请用中文输出 2~4 句极简纯文本复盘教训（严格控制在 80 tokens / 120 汉字以内；"
            "严禁使用 #、**、列表符号或换行标题）：\n"
            f"1. 点明 T+{holding_days} 收益与相对 {benchmark_symbol} 的超额 Alpha 表现；\n"
            "2. 指出原推荐逻辑中哪部分被验证或证伪；\n"
            "3. 提炼 1 条下次同类选股或择时可直接复用的具体纪律。\n"
            "反后视纪律：\n"
            "- 用 MFE/MAE 诊断是方向判断错误，还是止损过紧/止盈过早；\n"
            "- 若过程象限为 GOOD_LOSS，严禁建议放宽止损或扛单；\n"
            "- 若为 BAD_WIN，必须明确警示这是运气陷阱，不可复制；\n"
            "- 持仓期有客观催化时只引用已给出的事实，无催化不得编故事。"
        )
        prompt = (
            f"标的: {symbol} ({account} | {action})\n"
            f"收益口径: {direction_label}\n"
            f"推荐日期: {rec_date.isoformat()} -> 结算日: {exit_date.isoformat()} (T+{holding_days})\n"
            f"入场价: {entry_price:.4g} -> 结算价: {exit_price:.4g}\n"
            f"原推荐逻辑: {thesis}\n"
            f"真实收益率: {raw_return_pct:+.2f}%\n"
            f"基准({benchmark_symbol})同期: {bench_return_pct:+.2f}%\n"
            f"超额 Alpha: {alpha_pct:+.2f}%"
            + ("\n" + "\n".join(extra_lines) if extra_lines else "")
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

    async def _load_recent_fills_by_account(self) -> tuple[dict[str, list[Any]], set[str]]:
        """Best-effort T212 recent fills per account; failures degrade to "not checked" (None)."""
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:  # noqa: BLE001 - settlement must never fail on this lookup
            logger.debug("Settlement recent-fill portfolio fetch failed: %s", exc)
            return {}, set()
        fills: dict[str, list[Any]] = {}
        available: set[str] = set()
        for acc in (portfolio.isa, portfolio.invest):
            if not getattr(acc, "ok", False) or not getattr(acc, "order_history_supported", False):
                continue
            label = str(getattr(acc, "account_label", "") or "").strip()
            if not label:
                continue
            fills[label] = list(getattr(acc, "recent_orders", None) or [])
            available.add(label)
        return fills, available

    async def _run_daily_settlement_impl(self, now_utc: datetime | None = None) -> dict[str, Any]:
        base = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        dt_uk = base.astimezone(LONDON_TZ)
        uk_now_str = dt_uk.strftime("%Y-%m-%d %H:%M")
        # Read-only fill ledger: matching real fills never rewrites wallets, only annotates the pick.
        fills_by_account, fills_available = await self._load_recent_fills_by_account()
        # v5.5.10 Phase 2: one shared Flash Lite budget for the whole night (T+5/T+20 + EOD anomalies).
        search_budget = self._postmarket_search_budget()
        pending = await self.db.get_pending_settlements(limit=50, only_with_entry_price=False)
        followups = (
            await self.db.get_settled_recommendations_for_followup(limit=50)
            if hasattr(self.db, "get_settled_recommendations_for_followup")
            else []
        )
        # v5.5.9 M6: shadow counterfactual queue — hasattr guard keeps stub-DB tests
        # and older DB objects fully functional when the lane is absent.
        shadow_pending: list[dict[str, Any]] = []
        if hasattr(self.db, "get_pending_shadow_settlements"):
            try:
                shadow_pending = await self.db.get_pending_shadow_settlements(limit=50)
            except Exception as exc:
                logger.warning("Shadow settlement queue fetch failed: %s", exc)
                shadow_pending = []

        skipped = 0
        not_due = 0
        settled = 0
        not_filled = 0
        failed = 0
        followup_updated = 0
        shadow_settled = 0
        mature_items: list[tuple[dict[str, Any], date, date, str, int, float, list[str]]] = []
        followup_items: list[tuple[dict[str, Any], date, str, float, list[str], date | None, date | None]] = []
        shadow_items: list[tuple[dict[str, Any], date, str, float, list[str]]] = []

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

        # v5.5.9 M6: shadow counterfactual T+20 pre-filter — same trading-day due
        # convention as the live follow-up lane, but no fill reconciliation and
        # zero LLM reflection (the shadow trade never executed).
        for rec in shadow_pending:
            symbol = str(rec.get("symbol") or "").strip().upper()
            try:
                entry_val = float(rec.get("entry_price") or 0.0)
            except (TypeError, ValueError):
                continue
            if symbol in ("", "CASH") or not math.isfinite(entry_val) or entry_val <= 0:
                continue
            rec_date = _parse_recommendation_date(rec)
            market = infer_market_from_symbol(symbol)
            t20_date = add_trading_days(rec_date, 20, market=market)
            if not has_market_closed_on_date(t20_date, market=market, now_utc=base):
                continue
            shadow_items.append((rec, rec_date, market, entry_val, _resolve_benchmark_candidates(symbol, rec.get("benchmark_symbol"))))

        if mature_items or followup_items or shadow_items:
            fetch_symbols = list(
                dict.fromkeys(
                    [str(rec["symbol"]).strip().upper() for rec, *_ in mature_items]
                    + [b for *_, b_list in mature_items for b in b_list]
                    + [str(rec["symbol"]).strip().upper() for rec, *_ in followup_items]
                    + [b for _, _, _, _, b_list, _, _ in followup_items for b in b_list]
                    + [str(rec["symbol"]).strip().upper() for rec, *_ in shadow_items]
                    + [b for *_, b_list in shadow_items for b in b_list]
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
                # Shadow rows stay pending when the shared history batch fails —
                # settling them against empty OHLC would fabricate data.
                shadow_items = []

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

                action = str(rec.get("action") or "BUY").strip().upper()
                direction = _trade_direction(action)
                # A recommendation generated after that session closed (e.g. Friday 21:30
                # /pick) cannot have been filled at the same day's low; start next session.
                exclude_rec_date = False
                rec_ms = int(rec.get("created_at_ms") or 0)
                if rec_ms > 0:
                    rec_dt_utc = datetime.fromtimestamp(rec_ms / 1000.0, tz=timezone.utc)
                    exclude_rec_date = has_market_closed_on_date(
                        rec_date, market=market, now_utc=rec_dt_utc
                    )
                fill_status = (
                    "ASSUMED"
                    if str(rec.get("source") or "").strip().lower() == "report"
                    else _entry_fill_status(
                        action,
                        stock_ohlc,
                        rec_date,
                        target_date,
                        eff_entry,
                        exclude_rec_date=exclude_rec_date,
                    )
                )
                account_label = str(rec.get("account") or "").strip()
                account_key = (
                    account_label
                    if account_label in fills_by_account
                    else _normalize_account_label(account_label)
                )
                actual_filled: int | None = None
                actual_fill_price: float | None = None
                execution_slippage_pct: float | None = None
                if account_key in fills_available:
                    matched_fill = _match_recommendation_fill(
                        rec,
                        start_date=rec_date,
                        end_date=exit_date,
                        recent_orders=fills_by_account.get(account_key) or [],
                        entry_price=eff_entry,
                    )
                    if matched_fill is not None:
                        actual_filled = 1
                        actual_fill_price = round(float(matched_fill[0]), 4)
                        execution_slippage_pct = round(float(matched_fill[1]), 3)
                        fill_status = "FILLED"
                    else:
                        actual_filled = 0
                if fill_status == "NOT_FILLED":
                    saved = await self.db.settle_recommendation(
                        rec_id,
                        benchmark_symbol=bench_candidates[0] if bench_candidates else "SPY",
                        settled_at_uk=uk_now_str,
                        raw_return_pct=None,
                        alpha_pct=None,
                        outcome_status="NOT_FILLED",
                        reflection_lesson="",
                        fill_status="NOT_FILLED",
                        actual_filled=actual_filled,
                        actual_fill_price=actual_fill_price,
                        execution_slippage_pct=execution_slippage_pct,
                    )
                    if saved:
                        not_filled += 1
                    continue

                unadjusted_stock_ret = ((exit_price - eff_entry) / eff_entry) * 100.0
                raw_return_pct = unadjusted_stock_ret * direction
                bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, exit_date)
                if bench_res is None:
                    logger.warning("Benchmark history unavailable for settling %s (%s -> %s)", symbol, rec_date, exit_date)
                    failed += 1
                    continue
                chosen_bench, bench_ret_pct = bench_res
                # direction=-1 => benchmark outperformance of an avoided loser is positive alpha.
                alpha_pct = _direction_adjusted_alpha_pct(action, eff_entry, exit_price, bench_ret_pct)
                mfe_pct, mae_pct = _compute_mfe_mae_pct(
                    stock_ohlc, rec_date, exit_date, eff_entry, direction=direction
                )

                isa_value_entry = (
                    _normalize_account_label(account_label) == "ISA"
                    and action in {"BUY", "ADD"}
                    and holding_days <= 5
                )
                process_quadrant = classify_process_quadrant(
                    action=action,
                    raw_return_pct=raw_return_pct,
                    alpha_pct=alpha_pct,
                    fill_status=fill_status,
                    actual_filled=actual_filled,
                    execution_slippage_pct=execution_slippage_pct,
                )
                holding_catalyst = ""
                catalyst_needed = (
                    abs(float(raw_return_pct)) >= 8.0
                    or abs(float(alpha_pct)) >= 6.0
                    or (mae_pct is not None and float(mae_pct) <= -8.0)
                )
                if (
                    not isa_value_entry
                    and search_budget > 0
                    and catalyst_needed
                ):
                    search_budget -= 1
                    holding_catalyst = await self._fetch_catalyst_with_flash_lite(
                        symbol,
                        window_label=f"T+{holding_days} settlement ({rec_date.isoformat()} to {exit_date.isoformat()})",
                        reason_hint=(
                            f"T+{holding_days} raw return {raw_return_pct:+.1f}% | "
                            f"alpha {alpha_pct:+.1f}% | MAE {mae_pct if mae_pct is not None else 'N/A'}"
                        ),
                        db=self.db,
                        cache_scope=f"settle_t{holding_days}_{rec_date.isoformat()}",
                        uk_date=dt_uk.date().isoformat(),
                    )

                # v5.5.3 dual-philosophy layering: an ISA value BUY/ADD at T+5 is noise, not a lesson.
                # Keep the T+5 outcome numeric only; the real post-mortem lands on the T+20 follow-up.
                if isa_value_entry:
                    lesson = ""
                else:
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
                            direction=direction,
                            mfe_pct=mfe_pct,
                            mae_pct=mae_pct,
                            actual_filled=actual_filled,
                            actual_fill_price=actual_fill_price,
                            execution_slippage_pct=execution_slippage_pct,
                            regime=str(rec.get("regime") or ""),
                            process_quadrant=process_quadrant,
                            holding_catalyst=holding_catalyst,
                        )
                    except Exception as exc:
                        # v5.5.10 Phase 2: a transient LLM failure must not keep a numeric
                        # settlement PENDING forever; T+20 can still append the lesson later.
                        logger.warning(
                            "Settlement reflection failed for %s (id=%s), settling numerically: %s",
                            symbol,
                            rec_id,
                            exc,
                        )
                        lesson = ""

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
                    fill_status=fill_status,
                    actual_filled=actual_filled,
                    actual_fill_price=actual_fill_price,
                    execution_slippage_pct=execution_slippage_pct,
                    process_quadrant=process_quadrant,
                    holding_catalyst=holding_catalyst,
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
                action = str(rec.get("action") or "BUY").strip().upper()
                direction = _trade_direction(action)

                t10_ret: float | None = None
                t10_alpha: float | None = None
                t20_ret: float | None = None
                t20_alpha: float | None = None
                furthest_exit: date | None = None
                t20_ctx: tuple[str, float, float, date, float] | None = None

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
                    raw_ret = _direction_adjusted_return_pct(action, eff_entry, ex_price)
                    bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, ex_date)
                    if bench_res is None:
                        continue
                    b_sym, b_ret = bench_res
                    alpha_val = _direction_adjusted_alpha_pct(action, eff_entry, ex_price, b_ret)
                    if horizon_label == "t10":
                        t10_ret = round(raw_ret, 2)
                        t10_alpha = round(alpha_val, 2)
                    else:
                        t20_ret = round(raw_ret, 2)
                        t20_alpha = round(alpha_val, 2)
                        t20_ctx = (b_sym, b_ret, ex_price, ex_date, eff_entry)
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
                        mfe_new, mae_new = _compute_mfe_mae_pct(
                            stock_ohlc, rec_date, furthest_exit, eff_entry, direction=direction
                        )
                        prev_mfe = rec.get("mfe_pct")
                        prev_mae = rec.get("mae_pct")
                        if mfe_new is not None and prev_mfe is not None:
                            mfe_new = max(float(prev_mfe), mfe_new)
                        if mae_new is not None and prev_mae is not None:
                            mae_new = min(float(prev_mae), mae_new)

                t20_quadrant = ""
                if t20_alpha is not None:
                    t20_quadrant = classify_process_quadrant(
                        action=action,
                        raw_return_pct=t20_ret,
                        alpha_pct=t20_alpha,
                        fill_status=str(rec.get("fill_status") or ""),
                        actual_filled=rec.get("actual_filled"),
                        execution_slippage_pct=rec.get("execution_slippage_pct"),
                    )

                # T+20 补写：ISA 价值仓 T+5 刻意不写短期反思，在这里用 20 日真实表现补一条。
                t20_lesson: str | None = None
                t20_catalyst = ""
                if (
                    t20_alpha is not None
                    and t20_ctx is not None
                    and not str(rec.get("reflection_lesson") or "").strip()
                ):
                    b_sym, b_ret, t20_exit_price, t20_exit_date, t20_eff_entry = t20_ctx
                    catalyst_needed = (
                        abs(float(t20_ret or 0.0)) >= 8.0
                        or abs(float(t20_alpha)) >= 6.0
                        or (mae_new is not None and float(mae_new) <= -8.0)
                    )
                    if search_budget > 0 and catalyst_needed:
                        search_budget -= 1
                        t20_catalyst = await self._fetch_catalyst_with_flash_lite(
                            symbol,
                            window_label=f"T+20 follow-up ({rec_date.isoformat()} to {t20_exit_date.isoformat()})",
                            reason_hint=(
                                f"T+20 raw return {float(t20_ret or 0.0):+.1f}% | "
                                f"alpha {float(t20_alpha):+.1f}% | "
                                f"MAE {mae_new if mae_new is not None else 'N/A'}"
                            ),
                            db=self.db,
                            cache_scope=f"settle_t20_{rec_date.isoformat()}",
                            uk_date=dt_uk.date().isoformat(),
                        )
                    try:
                        t20_lesson = await self._generate_settlement_reflection(
                            rec=rec,
                            rec_date=rec_date,
                            exit_date=t20_exit_date,
                            holding_days=20,
                            entry_price=t20_eff_entry,
                            exit_price=t20_exit_price,
                            raw_return_pct=float(t20_ret or 0.0),
                            benchmark_symbol=b_sym,
                            bench_return_pct=b_ret,
                            alpha_pct=float(t20_alpha),
                            direction=direction,
                            mfe_pct=mfe_new,
                            mae_pct=mae_new,
                            actual_filled=rec.get("actual_filled"),
                            actual_fill_price=rec.get("actual_fill_price"),
                            execution_slippage_pct=rec.get("execution_slippage_pct"),
                            regime=str(rec.get("regime") or ""),
                            process_quadrant=t20_quadrant,
                            holding_catalyst=t20_catalyst,
                        )
                    except Exception as exc:
                        logger.warning("T+20 reflection failed for %s (id=%s): %s", symbol, rec_id, exc)
                        t20_lesson = None

                updated = await self.db.update_recommendation_followup(
                    rec_id,
                    t10_return_pct=t10_ret,
                    t10_alpha_pct=t10_alpha,
                    t20_return_pct=t20_ret,
                    t20_alpha_pct=t20_alpha,
                    mfe_pct=mfe_new,
                    mae_pct=mae_new,
                    reflection_lesson=t20_lesson,
                    process_quadrant=t20_quadrant or None,
                    holding_catalyst=t20_catalyst or None,
                )
                if updated:
                    followup_updated += 1

            # v5.5.9 M6: shadow counterfactual settlement (T+20, 0-token, no fills).
            # Consumes the same already-fetched history batch — zero extra requests.
            if shadow_items and hasattr(self.db, "settle_shadow_rejection"):
                for rec, rec_date, market, entry_price, bench_candidates in shadow_items:
                    rec_id = int(rec["id"])
                    symbol = str(rec["symbol"]).strip().upper()
                    stock_ohlc = _get_ohlc(symbol, market)
                    if stock_ohlc.empty:
                        continue
                    stock_closes = stock_ohlc["Close"]
                    t20_date = add_trading_days(rec_date, 20, market=market)
                    exit_dates = [d for d in stock_closes.index if d >= t20_date]
                    if not exit_dates:
                        continue
                    exit_date = exit_dates[0]
                    exit_price = float(stock_closes.loc[exit_date])
                    eff_entry = _align_entry_price_gbp_guard(symbol, entry_price, stock_closes, rec_date, exit_price)
                    if eff_entry is None:
                        logger.warning(
                            "Shadow settlement skipped for %s (id=%s): unresolved GBp/GBP 100x mismatch",
                            symbol,
                            rec_id,
                        )
                        continue
                    action = str(rec.get("action") or "BUY").strip().upper()
                    raw_return_pct = _direction_adjusted_return_pct(action, eff_entry, exit_price)
                    bench_res = _compute_benchmark_return_pct(bench_candidates, _get_closes, rec_date, exit_date)
                    if bench_res is None:
                        continue
                    chosen_bench, bench_ret_pct = bench_res
                    alpha_pct = _direction_adjusted_alpha_pct(action, eff_entry, exit_price, bench_ret_pct)
                    mfe_pct, mae_pct = _compute_mfe_mae_pct(
                        stock_ohlc, rec_date, exit_date, eff_entry, direction=_trade_direction(action)
                    )
                    saved = await self.db.settle_shadow_rejection(
                        rec_id,
                        benchmark_symbol=chosen_bench,
                        settled_at_uk=uk_now_str,
                        raw_return_pct=round(raw_return_pct, 2),
                        t20_return_pct=round(raw_return_pct, 2),
                        t20_alpha_pct=round(alpha_pct, 2),
                        mfe_pct=mfe_pct,
                        mae_pct=mae_pct,
                    )
                    if saved:
                        shadow_settled += 1

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

        nav_refreshed = ""
        dt_uk_now = base.astimezone(LONDON_TZ)
        if dt_uk_now.hour * 60 + dt_uk_now.minute >= NAV_SNAPSHOT_START_MINUTES:
            try:
                nav_refreshed = await self._run_nav_snapshot_job()
            except Exception as exc:
                logger.warning("Post-close NAV catch-up failed: %s", exc)

        # v5.5.10 Phase 2: 0-Token post-close execution ledger. Runs after the NAV
        # anchor and before the 90-day prune; every failure stays silent and never
        # blocks settlement or cleanup.
        postmarket_alert_text = ""
        if bool(getattr(self.cfg.strategy, "postmarket_review_enabled", True)) and dt_uk.weekday() < 5:
            try:
                postmarket_alert_text = await self._run_postmarket_review(
                    now_utc=base,
                    uk_date=dt_uk.date().isoformat(),
                    search_budget=search_budget,
                )
            except Exception as exc:
                logger.warning("Postmarket review failed: %s", exc)

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
            "not_filled": not_filled,
            "not_due": not_due,
            "failed": failed,
            "followup_updated": followup_updated,
            "shadow_settled": shadow_settled,
            "weekly_audit_ran": weekly_audit_ran,
            "nav_refreshed": nav_refreshed,
            "postmarket_alert_text": postmarket_alert_text,
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

    async def _run_nav_snapshot_job(self) -> str:
        """Persist one post-close GBP NAV row so the TWR curve samples EOD, not pre-market.

        Reuses the /report row when the day already exists (``uk_date`` is the primary key),
        so this simply refreshes it with closing values. Fallback portfolios are never
        written (iron rule 2).
        """
        portfolio = await self.t212.get_portfolio(use_cache=False)
        if portfolio is None or getattr(portfolio, "is_fallback", False):
            return "skipped: portfolio fallback"
        if float(getattr(portfolio, "total_combined_value", 0.0) or 0.0) <= 0.0:
            return "skipped: zero net value"

        bench_symbol = str(getattr(self.cfg.strategy, "benchmark_symbol", "VWRP.L") or "VWRP.L")
        bench_close: float | None = None
        try:
            hist_map = await self.market_data.get_histories([bench_symbol], period="1mo")
            bench_hist = hist_map.get(bench_symbol)
            if bench_hist is not None and not getattr(bench_hist, "empty", True):
                close_col = "Adj Close" if "Adj Close" in bench_hist.columns else "Close"
                if close_col in bench_hist.columns:
                    bench_close = float(bench_hist[close_col].dropna().iloc[-1])
        except Exception as exc:
            logger.debug("NAV snapshot benchmark close unavailable: %s", exc)

        row = build_nav_row(portfolio, benchmark_close=bench_close)
        today_uk = str(row.get("uk_date") or "")
        try:
            prev_row = await self.db.get_previous_nav_row(today_uk) if today_uk else None
        except Exception as exc:
            logger.debug("NAV previous row lookup skipped: %s", exc)
            prev_row = None
        if prev_row is not None:
            row = build_nav_row(
                portfolio,
                benchmark_close=bench_close,
                prev_snapshot_ms=int(prev_row.get("created_at_ms") or 0),
                prev_uk_date=str(prev_row.get("uk_date") or ""),
            )
        await self.db.upsert_nav_history(row)
        # v5.5.10: idempotently repair only the flow columns of the recent curve
        # (weekend / after-close omissions and sign flips). Never blocks the job.
        try:
            recent_rows = await self.db.get_nav_history(limit=35)
            updates = reconcile_recent_nav_flows(
                recent_rows,
                portfolio,
                history_days=int(getattr(self.cfg.strategy, "order_history_days", 30) or 30),
                max_pages=int(getattr(self.cfg.strategy, "order_history_max_pages", 3) or 3),
            )
            if updates:
                changed = await self.db.update_nav_history_flows(updates)
                logger.info("NAV flow reconciliation updated %d row(s)", changed)
        except Exception as exc:
            logger.debug("NAV flow reconciliation skipped: %s", exc)
        logger.info("NAV snapshot persisted: %s £%.2f", row.get("uk_date"), row.get("total_value_gbp", 0.0))
        return f"nav {row.get('uk_date')} £{row.get('total_value_gbp', 0.0):,.2f}"

    async def _run_daily_settlement_job(self, now_utc: datetime, *, alert_chat_id: str = "") -> str:
        summary = await self._run_daily_settlement_impl(now_utc=now_utc)
        alert_text = str(summary.get("postmarket_alert_text") or "").strip()
        if alert_text and alert_chat_id:
            try:
                await self.tg.send_message(alert_chat_id, alert_text)
            except Exception as exc:
                logger.warning("Postmarket alert push failed: %s", exc)
        pruned_total = sum(summary["pruned"].values())
        if summary["settled"] == 0 and summary["failed"] > 0:
            return (
                f"⚠️ 盘后结算异常：待处理 {summary['checked']} 条中 {summary['failed']} 条失败，"
                f"清理 {pruned_total} 条过期记录"
            )
        return (
            f"结算 {summary['settled']} 条 / 追评 {summary.get('followup_updated', 0)} / "
            f"未成交 {summary.get('not_filled', 0)} / 跳过 {summary['skipped']} / "
            f"未到期 {summary['not_due']} / 失败 {summary['failed']} / 清理 {pruned_total}"
            + (
                f" / 影子复盘 {summary.get('shadow_settled', 0)} 条"
                if summary.get("shadow_settled", 0) > 0
                else ""
            )
        )
