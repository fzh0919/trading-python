"""Saturday offline quant audit, benchmark dashboard, universe audition, and weekly wallet reconcile mixin."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.agents.ds_sidecar import call_ds_json, ds_helper_available, sanitize_plain_text
from src.bot.scheduler.settlement_math import LONDON_TZ
from src.data_providers.analytics import (
    _CORE_POLICY_BENCHMARK_WEIGHTS,
    compute_core_policy_benchmark_comparison,
    compute_window_return_pct,
)
from src.data_providers.t212_client import (
    AccountSnapshot,
    PortfolioData,
    compute_trade_behavior_stats,
)
from src.engine.screener import run_offline_universe_and_threshold_audit, run_universe_audition

logger = logging.getLogger("src.bot.scheduler")


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get("src.bot.scheduler")
    return getattr(facade, name, default) if facade is not None else default


class WeeklyAuditJobMixin:
    def _has_audit_universe(self) -> bool:
        return bool(
            self.cfg.universe.audition_enabled
            or self.cfg.universe.isa_long_term
            or self.cfg.universe.invest_tactical
            or self.cfg.universe.etf_symbols
        )

    def _build_benchmark_dashboard(
        self,
        portfolio: PortfolioData,
        histories: dict[str, pd.DataFrame],
        *,
        previous_snapshot: PortfolioData | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """VWRP 7d/30d benchmark returns vs account net value, realised P/L and fee drag."""
        symbol = self.cfg.strategy.benchmark_symbol
        hist = histories.get(symbol)
        if hist is None or getattr(hist, "empty", True):
            hist = histories.get(symbol.split(".")[0])
        ret_7 = compute_window_return_pct(hist, 5)  # 5 trading bars ≈ 7 calendar days
        ret_30 = compute_window_return_pct(hist, 21)  # 21 trading bars ≈ 30 calendar days

        now = int(now_ms if now_ms is not None else datetime.now(timezone.utc).timestamp() * 1000)
        behavior = compute_trade_behavior_stats(
            portfolio,
            recent_days=7,
            history_days=self.cfg.strategy.order_history_days,
            cooldown_days=self.cfg.strategy.rebuy_cooldown_days,
            now_ms=now,
        )
        realized_by_ccy: dict[str, float] = {}
        for o in (*portfolio.isa.recent_orders, *portfolio.invest.recent_orders):
            if (o.side or "").upper() != "SELL":
                continue
            ccy = str(o.wallet_currency or "GBP").strip().upper() or "GBP"
            realized_by_ccy[ccy] = realized_by_ccy.get(ccy, 0.0) + float(o.realised_pnl or 0.0)

        account_change_pct: float | None = None
        baseline_age_days: float | None = None
        if previous_snapshot is not None:
            prev_val = float(previous_snapshot.total_combined_value or 0.0)
            cur_val = float(portfolio.total_combined_value or 0.0)
            if prev_val > 0 and cur_val > 0:
                account_change_pct = ((cur_val - prev_val) / prev_val) * 100.0
            base_ms = int(previous_snapshot.fetched_at_ms or 0)
            if base_ms > 0:
                baseline_age_days = round((now - base_ms) / 86400000.0, 1)

        if ret_7 is not None and account_change_pct is not None:
            alpha = account_change_pct - ret_7
            verdict = f"近7天{'跑赢' if alpha >= 0 else '跑输'}基准 {alpha:+.1f}pct"
        elif ret_7 is not None:
            verdict = f"近7天基准 {ret_7:+.1f}%（缺少账户基线快照，暂无法算超额）"
        else:
            verdict = "基准行情数据不足"

        dashboard: dict[str, Any] = {
            "benchmark_symbol": symbol,
            "benchmark_ret_7d_pct": round(ret_7, 2) if ret_7 is not None else None,
            "benchmark_ret_30d_pct": round(ret_30, 2) if ret_30 is not None else None,
            "account_total_value_gbp": round(float(portfolio.total_combined_value or 0.0), 2),
            "account_change_pct": round(account_change_pct, 2) if account_change_pct is not None else None,
            "baseline_age_days": baseline_age_days,
            "realized_pnl_by_currency": {k: round(v, 2) for k, v in realized_by_ccy.items() if abs(v) >= 0.01},
            "fees_pct_of_turnover_30d": behavior.get("fees_pct_of_turnover"),
            "buys_7d": behavior.get("buys_7d"),
            "buys_30d": behavior.get("buys_30d"),
            "verdict": verdict,
        }
        # v5.5.3: surface the four-bucket core policy mix vs VWRP.L so the weekly review can
        # separate a deliberate UK/EU regional tilt (Beta) from genuine stock-picking Alpha.
        core_cmp = compute_core_policy_benchmark_comparison(histories, lookback_days=60)
        dashboard["core_policy_benchmark"] = core_cmp
        dashboard["summary_text"] = self._format_benchmark_dashboard_text(dashboard)
        return dashboard

    @staticmethod
    def _format_benchmark_dashboard_text(dash: dict[str, Any]) -> str:
        """Render the dashboard as one compact plain-text line (no markdown)."""
        symbol = dash.get("benchmark_symbol") or "VWRP.L"
        ret_7 = dash.get("benchmark_ret_7d_pct")
        ret_30 = dash.get("benchmark_ret_30d_pct")
        parts = [
            f"{symbol} 近7天 {ret_7:+.1f}%" if ret_7 is not None else f"{symbol} 近7天 N/A",
            f"近30天 {ret_30:+.1f}%" if ret_30 is not None else "近30天 N/A",
        ]
        change = dash.get("account_change_pct")
        if change is not None:
            parts.append(f"账户净值 {float(change):+.1f}%")
        realized = dash.get("realized_pnl_by_currency") or {}
        if realized:
            parts.append("已实现 " + " ".join(f"{ccy} {float(amt):+.0f}" for ccy, amt in sorted(realized.items())))
        fees = dash.get("fees_pct_of_turnover_30d")
        if fees is not None:
            parts.append(f"税费占成交额 {float(fees):.2f}%")
        core = dash.get("core_policy_benchmark") or {}
        if core:
            te = core.get("tracking_error_pct")
            parts.append(
                f"四桶核心{int(core.get('lookback_days') or 60)}日 组合 {float(core['policy_return_pct']):+.1f}% "
                f"vs {core['benchmark_symbol']} {float(core['benchmark_return_pct']):+.1f}%"
                f"（倾斜 {float(core['allocation_effect_pct']):+.1f}pct"
                + (f" | 跟踪误差 {float(te):.1f}%" if te is not None else "")
                + "）"
            )
        parts.append(str(dash.get("verdict") or ""))
        return " | ".join(p for p in parts if p)

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
        bench_dash = str((audit.get("benchmark_dashboard") or {}).get("summary_text") or "N/A")
        cov = audit.get("fundamentals_coverage") or {}
        cov_bits: list[str] = []
        for market, item in (cov.get("markets") or {}).items():
            if not item.get("symbols"):
                continue
            cov_map = item.get("coverage_pct") or {}
            cov_bits.append(
                f"{market}({item['symbols']}只) "
                + "/".join(f"{k} {v if v is not None else 'N/A'}" for k, v in list(cov_map.items())[:3])
            )
        cov_text = " | ".join(cov_bits) if cov_bits else "N/A"
        low_cov = "、".join((cov.get("low_coverage") or [])[:5]) or "无"
        try:
            quality = await self.db.get_ops_quality_summary(days=7)
        except Exception as exc:
            logger.debug("Weekly diagnosis ops quality summary failed: %s", exc)
            quality = None
        sem = (quality or {}).get("semantic_audit") or {}
        degrade_txt = (
            f"语义质检 {int(sem.get('total') or 0)} 次 | 降级 {int((quality or {}).get('degraded_alerts') or 0)} 次"
            if quality
            else "N/A"
        )
        lines = [
            f"【本周体检 {uk_date}】池 {audit.get('universe_count', 0)} 只 | "
            f"重叠 {len(audit.get('overlap_candidates') or [])} | "
            f"僵尸 {len(audit.get('zombie_candidates') or [])}",
            f"【基准对照原始数字】{bench_dash[:300]}",
            f"【价值因子覆盖率原始数字】{cov_text[:300]} | 覆盖率<50%: {low_cov[:120]}",
            f"【近7天质检与降级】{degrade_txt}",
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
                "你是量化策略诊断官（只读）。根据原始回测、T+5/10/20 后验与基准对照数字，输出 3 条以内中文诊断："
                "①相对 VWRP 基准是跑赢还是跑输及主因（仓位/择时/摩擦）②参数/阈值与胜率结构建议 ③下周关注。"
                "禁止编造未给出的数字，禁止 Markdown # 与 **，总长 <=450 字。"
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
                audition_res = await _facade_attr("run_universe_audition", run_universe_audition)(
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
        all_syms = list(
            dict.fromkeys(
                s for s in (held_syms + pool_syms + [self.cfg.strategy.benchmark_symbol]) if s
            )
        )

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
        # ISA value-factor coverage health: read the cached fundamentals (never fetch online here).
        if all_syms and hasattr(self.db, "get_fundamentals_cache"):
            try:
                fundamentals_cache = await self.db.get_fundamentals_cache(all_syms)
                audit_kwargs["fundamentals_rows"] = list((fundamentals_cache or {}).values())
            except Exception as exc:
                logger.debug("Weekly fundamentals coverage cache read failed: %s", exc)
        audit = run_offline_universe_and_threshold_audit(**audit_kwargs)
        if audition_res:
            audit["audition"] = audition_res
        previous_snapshot: PortfolioData | None = None
        try:
            previous_snapshot = await self.db.get_previous_snapshot(min_age_hours=24 * 6)
        except Exception as exc:
            logger.debug("Weekly benchmark baseline snapshot unavailable: %s", exc)
        # Make sure all four core-policy sleeves are present before the comparison runs.
        missing_core_syms = [s for s in _CORE_POLICY_BENCHMARK_WEIGHTS if s not in histories]
        if missing_core_syms:
            try:
                extra_hist = await self.market_data.get_histories(missing_core_syms, period="2y")
                histories.update(extra_hist or {})
            except Exception as exc:
                logger.debug("Weekly core policy histories fetch failed: %s", exc)
        audit["benchmark_dashboard"] = self._build_benchmark_dashboard(
            portfolio,
            histories,
            previous_snapshot=previous_snapshot,
            now_ms=int(base.timestamp() * 1000),
        )
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
                return await _facade_attr("run_universe_audition", run_universe_audition)(
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
        dash = str((audit.get("benchmark_dashboard") or {}).get("summary_text") or "").strip()
        benchmark_block = f"\n{dash}" if dash else ""
        if diagnosis:
            return (
                f"📊 周六策略体检周报（{audit.get('audit_date', '')}）\n"
                f"{diagnosis}{benchmark_block}\n\n{summary}"
            )
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
