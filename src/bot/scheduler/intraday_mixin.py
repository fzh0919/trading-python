"""Pre-market data prewarm, pending-order fill receipt collector, and dual-philosophy intraday sentinel mixin."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any

import pandas as pd

from src.agents.report_agent import generate_intraday_alert
from src.bot.scheduler.fill_receipts import (
    _drawdown_from_high_pct,
    _format_pending_orders,
    _format_uk_from_ms,
    _has_protective_stop,
    _intraday_move_pct,
    _invest_adaptive_stop_pct,
    _normalize_account_label,
    _order_distance_pct,
    _pending_order_snapshot_row,
    _snapshot_key,
    _to_float,
    diff_pending_order_snapshots,
    format_fill_receipt,
    match_fill_history,
)
from src.bot.scheduler.settlement_math import LONDON_TZ, _align_lse_unit_100x
from src.engine.calendar_utils import (
    get_us_market_schedule_uk,
    infer_market_from_symbol,
    is_eu_market_holiday,
    is_uk_bank_holiday,
)
from src.engine.deployment_gate import core_sleeve_for_symbol

logger = logging.getLogger("src.bot.scheduler")


class IntradaySentinelJobMixin:
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

    async def _collect_order_fill_receipts(self, now_utc: datetime, *, uk_date: str) -> list[tuple[str, str]]:
        """Diff the 0-token pending-order snapshot and return [(dedupe_key, receipt_text)].

        Reuses the portfolio already fetched by the intraday sentinel (cache-backed) and stays
        silent whenever either the pending-order or the order-history API is unavailable.
        """
        if not bool(getattr(self.cfg.strategy, "intraday_fill_receipt_enabled", True)):
            return []
        get_snapshots = getattr(self.db, "get_pending_order_snapshots", None)
        upsert_snapshots = getattr(self.db, "upsert_pending_order_snapshots", None)
        delete_snapshots = getattr(self.db, "delete_pending_order_snapshots", None)
        if not (callable(get_snapshots) and callable(upsert_snapshots) and callable(delete_snapshots)):
            return []
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Fill receipt portfolio fetch failed: %s", exc)
            return []

        accounts = [acc for acc in (portfolio.isa, portfolio.invest) if getattr(acc, "ok", False)]
        supported = {
            str(acc.account_label or "").strip()
            for acc in accounts
            if getattr(acc, "pending_orders_supported", False)
        }
        history_ok = any(getattr(acc, "order_history_supported", False) for acc in accounts)
        if not supported or not history_ok:
            # Cannot cross-verify a fill -> stay silent instead of risking a false receipt.
            return []
        history_accounts = {
            str(acc.account_label or "").strip()
            for acc in accounts
            if getattr(acc, "order_history_supported", False)
        }

        live_rows = [
            _pending_order_snapshot_row(order)
            for acc in accounts
            if str(acc.account_label or "").strip() in supported
            for order in (getattr(acc, "pending_orders", None) or [])
        ]
        try:
            previous = await get_snapshots()
        except Exception as exc:
            logger.warning("Fill receipt snapshot load failed: %s", exc)
            return []

        diff = diff_pending_order_snapshots(previous, live_rows, supported_accounts=supported)
        history = [*portfolio.isa.recent_orders, *portfolio.invest.recent_orders]
        now_ms = int(now_utc.timestamp() * 1000)
        receipts: list[tuple[str, str]] = []
        stale_keys: list[tuple[str, str]] = []
        claimed_fill_ids: set[str] = set()
        # Read-only USD wallet anchor for the receipt hint; a lookup failure must never block receipts.
        usd_wallet_available: float | None = None
        get_wallet_balances = getattr(self.db, "get_wallet_balances", None)
        if callable(get_wallet_balances):
            try:
                balances = await get_wallet_balances("Invest")
                usd_row = (balances or {}).get("USD") or {}
                usd_wallet_available = float(usd_row.get("available_amount") or 0.0)
            except Exception as exc:  # noqa: BLE001 - degrade silently to no hint
                logger.debug("Fill receipt USD wallet lookup failed: %s", exc)
                usd_wallet_available = None

        for row in diff["partial"]:
            key = _snapshot_key(row)
            dedupe = f"{key[0]}:{key[1]}:p{float(row.get('filled_quantity') or 0.0):g}"
            if await self._is_job_locked(f"fill_receipt:{dedupe}", uk_date):
                continue
            match = match_fill_history(row, history, excluded_fill_ids=claimed_fill_ids) or {
                "matched_quantity": float(row.get("filled_quantity") or 0.0),
                "avg_price": 0.0,
                "last_filled_at_ms": now_ms,
                "currency": str(row.get("currency") or ""),
                "is_complete": False,
            }
            claimed_fill_ids.update(match.get("fill_ids") or [])
            receipts.append((dedupe, format_fill_receipt(row, match, partial=True, usd_wallet_available=usd_wallet_available)))

        for row in diff["vanished"]:
            key = _snapshot_key(row)
            dedupe = f"{key[0]}:{key[1]}"
            if key[0] not in history_accounts:
                # No order history for this account this round -> cannot verify, keep the row.
                continue
            stale_keys.append(key)
            if await self._is_job_locked(f"fill_receipt:{dedupe}", uk_date):
                continue
            match = match_fill_history(row, history, excluded_fill_ids=claimed_fill_ids)
            if match is None:
                logger.info(
                    "Pending order %s %s vanished without a matching fill (likely cancelled)",
                    key[0],
                    key[1],
                )
                continue
            claimed_fill_ids.update(match.get("fill_ids") or [])
            cross_day = _format_uk_from_ms(int(match.get("last_filled_at_ms") or 0))[:10] != uk_date
            receipts.append(
                (
                    dedupe,
                    format_fill_receipt(
                        row,
                        match,
                        partial=not bool(match.get("is_complete", True)),
                        cross_day=cross_day,
                        usd_wallet_available=usd_wallet_available,
                    ),
                )
            )

        try:
            if live_rows:
                await upsert_snapshots(
                    live_rows,
                    seen_at_ms=now_ms,
                    seen_at_uk=now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
                )
            if stale_keys:
                await delete_snapshots(stale_keys)
        except Exception as exc:
            logger.warning("Fill receipt snapshot persist failed: %s", exc)
        return receipts

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
                receipts = await self._collect_order_fill_receipts(now_utc, uk_date=uk_date)
            except Exception:
                logger.exception("Fill receipt check crashed; skipping receipts this round")
                receipts = []
            try:
                alerts = await self._run_intraday_sentinel_impl(now_utc, uk_date=uk_date)
            except Exception:
                logger.exception("Intraday sentinel scan crashed; skipping this round")
                alerts = []
            finally:
                if self.current_job_name == "intraday_sentinel":
                    self.current_job_name = prev_job

        uk_now_str = now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
        pushed: list[str] = []
        for dedupe_key, text in receipts:
            try:
                delivered = await self.tg.send_message(chat_id, text)
            except Exception as exc:  # noqa: BLE001 - delivery failure must not break the loop
                logger.warning("Fill receipt delivery failed for %s: %s", dedupe_key, exc)
                continue
            if delivered is False:
                logger.warning("Fill receipt for %s was not delivered; keeping the dedupe slot free.", dedupe_key)
                continue
            # Mark only after a successful push so a failed delivery never mutes the receipt.
            await self._mark_job_locked(f"fill_receipt:{dedupe_key}", uk_date, uk_now_str)
            pushed.append(f"FILL:{dedupe_key}")

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

        position_by_symbol: dict[str, Any] = {}
        positions_by_symbol: dict[str, list[Any]] = {}
        for pos in portfolio.all_positions():
            sym = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            if not sym:
                continue
            positions_by_symbol.setdefault(sym, []).append(pos)
            position_by_symbol.setdefault(sym, pos)

        strategy = self.cfg.strategy
        invest_move_threshold = float(getattr(strategy, "intraday_move_alert_pct", 3.5) or 3.5)
        isa_move_threshold = float(getattr(strategy, "intraday_isa_move_alert_pct", 5.0) or 5.0)
        activation_pct = float(getattr(strategy, "trailing_profit_activation_pct", 20.0) or 20.0)
        trailing_dd_pct = float(getattr(strategy, "trailing_drawdown_from_peak_pct", -12.0) or -12.0)
        if trailing_dd_pct > 0:
            trailing_dd_pct = -trailing_dd_pct
        core_drawdown_steps = sorted(
            {abs(float(step)) for step in (getattr(strategy, "core_etf_drawdown_steps_pct", None) or []) if step},
            reverse=True,
        )

        today_uk = dt_uk.date()
        triggered: list[tuple[float, str, float, float, float | None, str, str, str]] = []
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

            positions = positions_by_symbol.get(sym) or []
            invest_positions = [
                p for p in positions if _normalize_account_label(getattr(p, "account", "")) == "Invest"
            ]
            isa_positions = [
                p for p in positions if _normalize_account_label(getattr(p, "account", "")) == "ISA"
            ]
            # Cross-account double holding follows the more sensitive Invest discipline.
            account_ctx = "Invest" if invest_positions else ("ISA" if isa_positions else "")
            move_threshold = invest_move_threshold if account_ctx != "ISA" else isa_move_threshold
            hit_move = move_pct is not None and abs(move_pct) >= move_threshold
            hit_order = order_dist is not None and order_dist <= strategy.intraday_order_near_pct

            reasons: list[str] = []
            priority = abs(move_pct or 0.0)
            if hit_move:
                direction = "上涨" if float(move_pct or 0.0) > 0 else "下跌"
                reasons.append(
                    f"日内{direction} {abs(float(move_pct or 0.0)):.1f}%（{account_ctx or '挂单'}阈值 {move_threshold:.1f}%）"
                )
            if hit_order:
                reasons.append(f"距最近挂单价 {float(order_dist or 0.0):.1f}%")

            if invest_positions:
                invest_pos = invest_positions[0]
                avg_price = _to_float(getattr(invest_pos, "avg_price", None))
                loss_pct = _to_float(getattr(invest_pos, "return_pct", None))
                cost_note = ""
                if avg_price is not None and avg_price > 0:
                    aligned_px = _align_lse_unit_100x(price, avg_price, sym) if sym.endswith(".L") else price
                    loss_pct = (aligned_px / avg_price - 1.0) * 100.0
                    cost_note = f"，成本 {avg_price:,.2f}"
                naked_invest = not _has_protective_stop(pending_by_symbol.get(sym), account="Invest")
                if naked_invest and loss_pct is not None:
                    stop_line = _invest_adaptive_stop_pct(histories.get(sym), strategy)
                    if loss_pct <= stop_line:
                        reasons.append(
                            "裸露止损/移动止盈逼近：Invest 无 STOP/STOP_LIMIT 挂单，"
                            f"浮亏 {loss_pct:.1f}%{cost_note} 已跌破自适应止损线 {stop_line:.1f}%"
                        )
                        priority = max(priority, abs(loss_pct))
                    elif loss_pct >= activation_pct:
                        dd60 = _drawdown_from_high_pct(histories.get(sym), window=60)
                        if dd60 is not None and dd60 <= trailing_dd_pct:
                            reasons.append(
                                "裸露止损/移动止盈逼近：Invest 无 STOP/STOP_LIMIT 挂单，"
                                f"浮盈 {loss_pct:.1f}%{cost_note} 后自近 60 日高点回撤 {dd60:.1f}%"
                            )
                            priority = max(priority, abs(dd60))

            if account_ctx == "ISA" and core_drawdown_steps:
                if core_sleeve_for_symbol(sym, bundles) is not None:
                    dd52 = _drawdown_from_high_pct(histories.get(sym), window=252)
                    if dd52 is not None:
                        crossed_steps = [step for step in core_drawdown_steps if dd52 <= -step]
                        if crossed_steps:
                            reasons.append(
                                f"ISA 核心宽基距 52 周高点回撤 {dd52:.1f}%，已触及 {crossed_steps[0]:.0f}% 阶梯"
                            )
                            priority = max(priority, abs(dd52))

            if not reasons:
                continue
            currency = str(
                getattr(bundle, "currency", "") or getattr(position, "inst_currency", "") or ""
            )
            triggered.append(
                (
                    priority,
                    sym,
                    price,
                    float(move_pct or 0.0),
                    order_dist,
                    currency,
                    account_ctx,
                    "；".join(reasons),
                )
            )

        triggered.sort(key=lambda item: item[0], reverse=True)
        max_alerts = 2
        alerts: list[tuple[str, str]] = []
        for _, sym, price, move_pct, order_dist, currency, account_ctx, trigger_reason in triggered:
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
                    account_context=account_ctx,
                    trigger_reason=trigger_reason,
                    ops_notifier=self.ops,
                )
            except Exception as exc:  # noqa: BLE001 - one symbol must never break the scan
                logger.warning("Intraday alert generation failed for %s: %s", sym, exc)
                text = None
            if text:
                alerts.append((sym, text))
        return alerts
