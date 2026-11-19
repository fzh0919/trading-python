"""Async Trading 212 API v0 client: metadata sync, single-account fetch/fallback, and portfolio aggregation."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from src.data_providers.t212_client.http_mixin import T212HttpMixin
from src.data_providers.t212_client.models import (
    AccountSnapshot,
    PortfolioData,
    PositionItem,
)
from src.data_providers.t212_client.wallets_and_tickers import (
    LONDON_TZ,
    _is_transient_http_status,
    _parse_ratelimit_remaining,
    _parse_ratelimit_reset_ts,
    clean_t212_ticker,
    reconcile_invest_wallets,
    to_yahoo_symbol,
)

logger = logging.getLogger("src.data_providers.t212_client")


class T212Client(T212HttpMixin):
    """Async client for Trading 212 API v0 with connection pooling, rate-limit header pacing, and fallback."""

    def __init__(
        self,
        base_url: str = "https://live.trading212.com/api/v0",
        isa_key: str = "",
        isa_secret: str = "",
        invest_key: str = "",
        invest_secret: str = "",
        order_history_days: int = 30,
        order_history_max_pages: int = 3,
        cash_interest_rate_pct: float = 4.0,
        usd_cash_interest_rate_pct: float = 3.4,
        invest_multi_currency_enabled: bool = True,
        wallet_reconcile_warn_days: int = 7,
        wallet_mismatch_warn_gbp: float = 10.0,
        cache_ttl_seconds: int = 120,
        transport: httpx.AsyncBaseTransport | None = None,
        ops_notifier: Any = None,
        snapshot_loader: Callable[[], Awaitable[PortfolioData | None]] | None = None,
        transient_backoff_s: float = 0.5,
        db: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.isa_key = isa_key.strip()
        self.isa_secret = isa_secret.strip()
        self.invest_key = invest_key.strip()
        self.invest_secret = invest_secret.strip()
        self.order_history_days = order_history_days
        self.order_history_max_pages = order_history_max_pages
        self.cash_interest_rate_pct = cash_interest_rate_pct
        self.usd_cash_interest_rate_pct = usd_cash_interest_rate_pct
        self.invest_multi_currency_enabled = bool(invest_multi_currency_enabled)
        self.wallet_reconcile_warn_days = int(wallet_reconcile_warn_days)
        self.wallet_mismatch_warn_gbp = float(wallet_mismatch_warn_gbp)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._transport = transport
        self._ops = ops_notifier
        self._snapshot_loader = snapshot_loader
        self._transient_backoff_s = max(0.0, float(transient_backoff_s))
        self._db = db
        self._client: httpx.AsyncClient | None = None
        self._portfolio_lock = asyncio.Lock()
        self._cache: tuple[float, PortfolioData] | None = None
        self._last_good_portfolio: PortfolioData | None = None
        self._last_good_accounts: dict[str, tuple[str, int, AccountSnapshot]] = {}
        self._wallet_records: dict[str, dict[str, Any]] = {}
        self._gbp_usd_rate: float | None = None
        self._rate_limit_resets: dict[tuple[str, str], float] = {}
        self._instrument_map: dict[str, dict[str, Any]] = {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def set_ops_notifier(self, ops_notifier: Any) -> None:
        self._ops = ops_notifier

    def set_snapshot_loader(self, snapshot_loader: Callable[[], Awaitable[PortfolioData | None]] | None) -> None:
        self._snapshot_loader = snapshot_loader

    def set_db(self, db: Any) -> None:
        self._db = db

    def set_gbp_usd_rate(self, gbp_usd_rate: float | None) -> None:
        """Inject the latest GBP/USD mid rate so Invest wallet splits can be derived.

        The first known rate invalidates the TTL cache so the next fetch upgrades
        USD-only wallet fields into a full GBP+USD split.
        """
        try:
            rate = float(gbp_usd_rate) if gbp_usd_rate is not None else 0.0
        except (TypeError, ValueError):
            return
        if rate <= 0.0:
            return
        if self._gbp_usd_rate is None:
            self._cache = None
        self._gbp_usd_rate = rate

    @property
    def gbp_usd_rate(self) -> float | None:
        """Latest injected GBP/USD mid rate used for Invest wallet derivation."""
        return self._gbp_usd_rate

    def _resolve_ticker_metadata(
        self,
        raw_ticker: str,
        inst: dict[str, Any],
        default_currency: str,
        extra_currency: str = "",
    ) -> tuple[str, str, str, str | None]:
        """Resolve (clean_ticker, inst_currency, yahoo_symbol, isin) using cached T212 instrument metadata."""
        clean = clean_t212_ticker(raw_ticker)
        meta = self._instrument_map.get(raw_ticker) or {}
        inst_curr = str(
            inst.get("currency")
            or meta.get("currency_code")
            or ("USD" if raw_ticker.upper().endswith("_US_EQ") else "")
            or extra_currency
            or default_currency
        )
        yahoo_sym = str(meta.get("yahoo_symbol") or "").strip() or to_yahoo_symbol(raw_ticker, inst_curr, clean)
        raw_isin = inst.get("isin") or meta.get("isin") or ""
        isin = str(raw_isin).strip().upper() or None
        return clean, inst_curr, yahoo_sym, isin

    async def fetch_instruments_metadata(self, *, account_scope: str | None = None) -> list[dict[str, Any]]:
        """Fetch /equity/metadata/instruments (1 req / 50s bucket) and sync to SQLite archive if available."""
        scope = str(account_scope or "").strip().upper() or None
        if scope == "ISA":
            if not (self.isa_key and self.isa_secret):
                return []
            api_key, api_secret = self.isa_key, self.isa_secret
        elif scope == "INVEST":
            if not (self.invest_key and self.invest_secret):
                return []
            api_key, api_secret = self.invest_key, self.invest_secret
        else:
            if self.isa_key and self.isa_secret:
                api_key, api_secret = self.isa_key, self.isa_secret
            elif self.invest_key and self.invest_secret:
                api_key, api_secret = self.invest_key, self.invest_secret
            else:
                return []

        client = self._get_client()
        headers = self._auth_headers(api_key, api_secret)
        url = f"{self.base_url}/equity/metadata/instruments"
        try:
            resp = await self._request_with_retry(
                client,
                url,
                headers,
                default_retry_sleep_s=50.5,
                max_retries=2,
                rate_bucket="metadata_instruments",
                max_rate_wait_s=55.0,
            )
        except httpx.HTTPError as exc:
            logger.warning("T212 /equity/metadata/instruments failed: %s", exc)
            return []

        remaining = _parse_ratelimit_remaining(resp.headers)
        reset_ts = _parse_ratelimit_reset_ts(resp.headers)
        if remaining is not None and remaining <= 0 and reset_ts is None:
            self._rate_limit_resets[("metadata_instruments", urlparse(url).path)] = time.time() + 50.0
        elif scope is not None:
            # Proactively enforce T212 1 req / 50s cooldown before any subsequent account-scope call
            self._rate_limit_resets[("metadata_instruments", urlparse(url).path)] = time.time() + 52.0

        if resp.status_code != 200:
            logger.warning("T212 /equity/metadata/instruments returned HTTP %d", resp.status_code)
            return []

        raw_items = resp.json()
        if not isinstance(raw_items, list):
            return []

        now_ms = int(time.time() * 1000)
        parsed_items: list[dict[str, Any]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            ticker = str(raw.get("ticker") or "").strip()
            if not ticker:
                continue
            currency_code = str(raw.get("currencyCode") or raw.get("currency") or "").strip()
            short_name = str(raw.get("shortName") or raw.get("name") or "").strip()
            isin = str(raw.get("isin") or "").strip().upper()
            inst_type = str(raw.get("type") or "").strip().upper()
            extended_hours = bool(raw.get("extendedHours", False))
            raw_sched = raw.get("workingScheduleId")
            working_schedule_id: int | None = None
            if raw_sched is not None and str(raw_sched).strip() != "":
                try:
                    working_schedule_id = int(raw_sched)
                except (TypeError, ValueError):
                    working_schedule_id = None
            clean_hint = short_name if re.fullmatch(r"[A-Za-z0-9._-]{1,10}", short_name) else None
            yahoo_sym = to_yahoo_symbol(ticker, currency_code or "USD", clean_ticker=clean_hint)
            item = {
                "ticker": ticker,
                "yahoo_symbol": yahoo_sym,
                "isin": isin,
                "currency_code": currency_code,
                "short_name": short_name,
                "inst_type": inst_type,
                "extended_hours": extended_hours,
                "working_schedule_id": working_schedule_id,
                "updated_at_ms": now_ms,
            }
            self._instrument_map[ticker] = item
            parsed_items.append(item)

        if parsed_items and self._db is not None and hasattr(self._db, "upsert_t212_instruments"):
            try:
                if scope is not None:
                    await self._db.upsert_t212_instruments(
                        parsed_items,
                        account_scope=scope,
                        full_snapshot=True,
                    )
                else:
                    await self._db.upsert_t212_instruments(parsed_items)
            except Exception as exc:
                logger.warning("Failed persisting T212 instrument metadata: %s", exc)
        return parsed_items

    async def sync_all_accounts_instruments_metadata(self) -> dict[str, int]:
        """Sync both ISA & Invest instrument lists sequentially, respecting T212's 1 req / 50s bucket."""
        isa_count = 0
        invest_count = 0
        if self.isa_key and self.isa_secret:
            logger.info("T212 metadata sync: fetching ISA instruments list...")
            isa_items = await self.fetch_instruments_metadata(account_scope="ISA")
            isa_count = len(isa_items)
            logger.info("T212 metadata sync: ISA fetched %d instruments.", isa_count)
        if self.invest_key and self.invest_secret:
            if isa_count > 0:
                logger.info("T212 metadata sync: waiting 52s cooldown before Invest instruments list (1 req/50s)...")
            invest_items = await self.fetch_instruments_metadata(account_scope="INVEST")
            invest_count = len(invest_items)
            logger.info("T212 metadata sync: Invest fetched %d instruments.", invest_count)
        return {"isa_count": isa_count, "invest_count": invest_count}

    async def fetch_single_account(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        api_secret: str,
        account_label: str,
    ) -> AccountSnapshot:
        snap, _ = await self._fetch_single_account_with_status(client, api_key, api_secret, account_label)
        return snap

    async def _fetch_single_account_with_status(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        api_secret: str,
        account_label: str,
    ) -> tuple[AccountSnapshot, bool]:
        """Fetch one account and return (snapshot, is_transient_error)."""
        if not api_key or not api_secret:
            return (
                AccountSnapshot(ok=False, account_label=account_label, error=f"{account_label} API credentials not configured."),
                False,
            )

        headers = self._auth_headers(api_key, api_secret)
        account_currency = "GBP"
        available_to_trade = 0.0
        in_pies = 0.0
        reserved = 0.0
        total_cash = 0.0
        cash_note = ""
        summary_inv_val: float | None = None
        summary_inv_cost: float | None = None
        summary_unrealized: float | None = None
        summary_realized: float | None = None
        summary_total_val: float | None = None
        summary_transient_err: str | None = None
        auth_alert_sent = False

        try:
            # 1. Account summary
            try:
                summary_res = await self._request_with_retry(
                    client,
                    f"{self.base_url}/equity/account/summary",
                    headers,
                    default_retry_sleep_s=5.2,
                    rate_bucket=account_label,
                )
                if summary_res.status_code == 200:
                    summary = summary_res.json() or {}
                    account_currency = str(summary.get("currency") or "GBP")
                    cash_obj = summary.get("cash") or {}
                    available_to_trade = float(cash_obj.get("availableToTrade") or 0)
                    in_pies = float(cash_obj.get("inPies") or 0)
                    reserved = float(cash_obj.get("reservedForOrders") or 0)
                    total_cash = available_to_trade + in_pies + reserved

                    inv = summary.get("investments") or {}
                    if inv.get("currentValue") is not None:
                        summary_inv_val = float(inv["currentValue"])
                    if inv.get("totalCost") is not None:
                        summary_inv_cost = float(inv["totalCost"])
                    if inv.get("unrealizedProfitLoss") is not None:
                        summary_unrealized = float(inv["unrealizedProfitLoss"])
                    if inv.get("realizedProfitLoss") is not None:
                        summary_realized = float(inv["realizedProfitLoss"])
                    if summary.get("totalValue") is not None:
                        summary_total_val = float(summary["totalValue"])
                else:
                    cash_note = f"(Account summary HTTP {summary_res.status_code})"
                    if summary_res.status_code in (401, 403):
                        auth_alert_sent = True
                        await self._notify_ops(
                            "CRITICAL",
                            f"t212_auth_failed:{account_label}",
                            f"Trading 212 [{account_label}] /equity/account/summary 返回 HTTP {summary_res.status_code}，API Key 鉴权失效。",
                            title=f"T212 {account_label} 鉴权失效",
                            business_blocking=True,
                            bot_message=f"🚨 Trading 212 [{account_label}] API Key 鉴权失效（HTTP {summary_res.status_code}），请及时更新只读密钥。",
                        )
                    if _is_transient_http_status(summary_res.status_code):
                        summary_transient_err = f"HTTP {summary_res.status_code}"
            except httpx.HTTPError as exc:
                summary_transient_err = f"{type(exc).__name__}: {exc}"
                cash_note = f"(Account summary {type(exc).__name__})"

            # 2. Positions
            pos_res = await self._request_with_retry(
                client,
                f"{self.base_url}/equity/positions",
                headers,
                default_retry_sleep_s=1.5,
                rate_bucket=account_label,
            )
            if pos_res.status_code != 200:
                if pos_res.status_code in (401, 403) and not auth_alert_sent:
                    await self._notify_ops(
                        "CRITICAL",
                        f"t212_auth_failed:{account_label}",
                        f"Trading 212 [{account_label}] /equity/positions 返回 HTTP {pos_res.status_code}，API Key 鉴权失效。",
                        title=f"T212 {account_label} 鉴权失效",
                        business_blocking=True,
                        bot_message=f"🚨 Trading 212 [{account_label}] API Key 鉴权失效（HTTP {pos_res.status_code}），今日风控与持仓同步已受阻。",
                    )
                return (
                    AccountSnapshot(
                        ok=False,
                        account_label=account_label,
                        error=f"{account_label} positions API HTTP {pos_res.status_code}: {pos_res.text[:200]}",
                    ),
                    _is_transient_http_status(pos_res.status_code),
                )

            raw_positions = pos_res.json() or []
            parsed_positions: list[PositionItem] = []
            calc_val = 0.0
            calc_cost = 0.0
            calc_pnl = 0.0
            total_fx_impact = 0.0

            for p in raw_positions:
                inst = p.get("instrument") or {}
                wallet = p.get("walletImpact") or {}
                raw_ticker = str(inst.get("ticker") or p.get("ticker") or "UNKNOWN")
                if wallet.get("currency"):
                    account_currency = str(wallet["currency"])
                clean, inst_curr, yahoo_sym, isin = self._resolve_ticker_metadata(
                    raw_ticker,
                    inst,
                    account_currency,
                )

                qty = float(p.get("quantity") or 0)
                avg_price = float(p.get("averagePricePaid") or 0)
                cur_price = float(p.get("currentPrice") or 0)

                gbx_scale = 0.01 if inst_curr in {"GBX", "GBp"} else 1.0
                current_value = (
                    float(wallet["currentValue"])
                    if wallet.get("currentValue") is not None
                    else cur_price * qty * gbx_scale
                )
                total_cost = (
                    float(wallet["totalCost"])
                    if wallet.get("totalCost") is not None
                    else avg_price * qty * gbx_scale
                )
                pnl = (
                    float(wallet["unrealizedProfitLoss"])
                    if wallet.get("unrealizedProfitLoss") is not None
                    else (current_value - total_cost)
                )
                fx_impact = float(wallet["fxImpact"]) if wallet.get("fxImpact") is not None else 0.0
                ret_pct = ((pnl / total_cost) * 100.0) if total_cost > 0 else 0.0

                raw_created_at = p.get("createdAt")
                created_at = str(raw_created_at).strip() if raw_created_at else None
                if created_at == "":
                    created_at = None

                calc_val += current_value
                calc_cost += total_cost
                calc_pnl += pnl
                total_fx_impact += fx_impact

                parsed_positions.append(
                    PositionItem(
                        account=account_label,
                        raw_ticker=raw_ticker,
                        ticker=clean,
                        yahoo_symbol=yahoo_sym,
                        name=str(inst.get("name") or clean),
                        inst_currency=inst_curr,
                        quantity=qty,
                        avg_price=avg_price,
                        cur_price=cur_price,
                        current_value=current_value,
                        total_cost=total_cost,
                        pnl=pnl,
                        fx_impact=fx_impact,
                        return_pct=ret_pct,
                        created_at=created_at,
                        isin=isin,
                    )
                )

            parsed_positions.sort(key=lambda x: x.current_value, reverse=True)

            # 3. Orders & Cash Events (Transactions / Dividends)
            pending_orders, pending_supported, pending_error, recent_orders, history_supported = await self._fetch_order_data(
                client, headers, account_label, account_currency
            )
            recent_transactions, tx_supported, recent_dividends, div_supported = await self._fetch_cash_events(
                client, headers, account_label, account_currency
            )

            # If summary had a transient error while positions succeeded, preserve cached cash if available
            if summary_transient_err is not None:
                mem_entry = self._last_good_accounts.get(account_label)
                if mem_entry is not None:
                    mem_uk, _, mem_acc = mem_entry
                    available_to_trade = mem_acc.available_to_trade
                    in_pies = mem_acc.in_pies_cash
                    reserved = mem_acc.reserved_cash
                    total_cash = mem_acc.total_cash
                    if summary_realized is None:
                        summary_realized = mem_acc.realized_pnl
                    cash_note = f"(摘要接口 {summary_transient_err}，现金沿用内存缓存 @ {mem_uk} UK)"

            investments_value = summary_inv_val if summary_inv_val is not None else calc_val
            investments_cost = summary_inv_cost if summary_inv_cost is not None else calc_cost
            unrealized_pnl = summary_unrealized if summary_unrealized is not None else calc_pnl
            total_account_value = summary_total_val if summary_total_val is not None else (investments_value + total_cash)
            cash_ratio = ((total_cash / total_account_value) * 100.0) if total_account_value > 0 else 0.0

            for pos in parsed_positions:
                pos.weight_pct = ((pos.current_value / total_account_value) * 100.0) if total_account_value > 0 else 0.0

            snap = AccountSnapshot(
                ok=True,
                account_label=account_label,
                account_currency=account_currency,
                total_account_value=total_account_value,
                investments_value=investments_value,
                investments_cost=investments_cost,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=summary_realized,
                total_cash=total_cash,
                available_to_trade=available_to_trade,
                reserved_cash=reserved,
                in_pies_cash=in_pies,
                cash_ratio_pct=cash_ratio,
                total_fx_impact=total_fx_impact,
                positions=parsed_positions,
                pending_orders=pending_orders,
                recent_orders=recent_orders,
                recent_transactions=recent_transactions,
                recent_dividends=recent_dividends,
                pending_orders_supported=pending_supported,
                pending_orders_error=pending_error,
                order_history_supported=history_supported,
                transactions_supported=tx_supported,
                dividends_supported=div_supported,
                cash_status_note=cash_note,
            )
            self._enrich_account_dividends(snap, None)
            return snap, False
        except httpx.HTTPError as exc:
            logger.warning("Transient HTTP error fetching %s account: %s", account_label, exc)
            return AccountSnapshot(ok=False, account_label=account_label, error=f"{type(exc).__name__}: {exc}"), True
        except Exception as exc:
            logger.exception("Error fetching %s account", account_label)
            return AccountSnapshot(ok=False, account_label=account_label, error=str(exc)), False

    @staticmethod
    def _annotate_account_fallback(
        acc: AccountSnapshot,
        source: str,
        orig_uk: str,
        reason: str,
    ) -> AccountSnapshot:
        fb = acc.model_copy(deep=True)
        src_label = "内存缓存" if source == "memory" else "数据库快照"
        tag = f"[⚠️ 5xx/断网回退{src_label} @ {orig_uk} UK]"
        fb.is_fallback = True
        fb.fallback_source = source
        fb.fallback_reason = reason[:160]
        fb.snapshot_fetched_at_uk = orig_uk
        fb.cash_status_note = f"{fb.cash_status_note} {tag}".strip() if fb.cash_status_note else tag
        return fb

    async def _resolve_account_fallback(
        self,
        account_label: str,
        reason: str,
        db_snapshot_cache: list[PortfolioData | None],
    ) -> tuple[AccountSnapshot, str, int, str] | None:
        """Try memory cache first, then optional DB snapshot_loader, for a failed account."""
        mem_entry = self._last_good_accounts.get(account_label)
        if mem_entry is not None:
            orig_uk, orig_ms, mem_acc = mem_entry
            return (
                self._annotate_account_fallback(mem_acc, "memory", orig_uk, reason),
                orig_uk,
                orig_ms,
                "memory",
            )

        if self._last_good_portfolio is not None:
            acc = self._last_good_portfolio.isa if account_label == "ISA" else self._last_good_portfolio.invest
            if acc.ok:
                orig_uk = acc.snapshot_fetched_at_uk or self._last_good_portfolio.fetched_at_uk
                orig_ms = self._last_good_portfolio.fetched_at_ms
                return (
                    self._annotate_account_fallback(acc, "memory", orig_uk, reason),
                    orig_uk,
                    orig_ms,
                    "memory",
                )

        if self._snapshot_loader is not None:
            if not db_snapshot_cache:
                try:
                    loaded = await self._snapshot_loader()
                except Exception as exc:
                    logger.warning("T212 snapshot_loader failed: %s", exc)
                    loaded = None
                db_snapshot_cache.append(loaded)
            db_snap = db_snapshot_cache[0]
            if db_snap is not None:
                acc = db_snap.isa if account_label == "ISA" else db_snap.invest
                if acc.ok:
                    orig_uk = acc.snapshot_fetched_at_uk or db_snap.fetched_at_uk
                    orig_ms = db_snap.fetched_at_ms
                    return (
                        self._annotate_account_fallback(acc, "db", orig_uk, reason),
                        orig_uk,
                        orig_ms,
                        "db",
                    )
        return None

    @staticmethod
    def _enrich_account_dividends(
        acc: AccountSnapshot,
        cum_map: dict[tuple[str, str], float] | None = None,
    ) -> None:
        """Populate cumulative_dividends_gbp & total_return_with_div_pct from DB archive and recent dividends."""
        if not acc.ok or not acc.positions:
            return
        recent_by_sym: dict[str, float] = {}
        for d in acc.recent_dividends:
            if d.yahoo_symbol and d.amount > 0:
                recent_by_sym[d.yahoo_symbol] = recent_by_sym.get(d.yahoo_symbol, 0.0) + float(d.amount)

        for pos in acc.positions:
            db_div = (cum_map or {}).get((acc.account_label, pos.yahoo_symbol), 0.0)
            rec_div = recent_by_sym.get(pos.yahoo_symbol, 0.0)
            cum_div = db_div if db_div > 0 else rec_div
            if cum_div > 0:
                pos.cumulative_dividends_gbp = round(float(cum_div), 4)
                if pos.total_cost > 0:
                    pos.total_return_with_div_pct = round(
                        ((pos.pnl + pos.cumulative_dividends_gbp) / pos.total_cost) * 100.0,
                        2,
                    )

    async def get_portfolio(self, use_cache: bool = True, gbp_usd_rate: float | None = None) -> PortfolioData:
        if gbp_usd_rate is not None:
            self.set_gbp_usd_rate(gbp_usd_rate)
        now = time.monotonic()
        if use_cache and self._cache is not None:
            cached_at, cached_data = self._cache
            if (now - cached_at) < self.cache_ttl_seconds:
                return cached_data

        async with self._portfolio_lock:
            now = time.monotonic()
            if use_cache and self._cache is not None:
                cached_at, cached_data = self._cache
                if (now - cached_at) < self.cache_ttl_seconds:
                    return cached_data

            if not self._instrument_map and self._db is not None and hasattr(self._db, "get_t212_instrument_map"):
                try:
                    self._instrument_map = await self._db.get_t212_instrument_map()
                except Exception as exc:
                    logger.debug("Failed loading cached T212 instrument map: %s", exc)

            client = self._get_client()
            (isa_snap, isa_transient), (invest_snap, invest_transient) = await asyncio.gather(
                self._fetch_single_account_with_status(client, self.isa_key, self.isa_secret, "ISA"),
                self._fetch_single_account_with_status(client, self.invest_key, self.invest_secret, "Invest"),
            )

            now_uk = datetime.now(timezone.utc).astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M")
            now_ms = int(time.time() * 1000)

            if isa_snap.ok:
                self._last_good_accounts["ISA"] = (now_uk, now_ms, isa_snap.model_copy(deep=True))
            if invest_snap.ok:
                self._last_good_accounts["Invest"] = (now_uk, now_ms, invest_snap.model_copy(deep=True))

            db_snapshot_cache: list[PortfolioData | None] = []
            fallback_meta: dict[str, tuple[str, int, str, str]] = {}

            if not isa_snap.ok and isa_transient:
                orig_err = isa_snap.error
                resolved = await self._resolve_account_fallback("ISA", orig_err, db_snapshot_cache)
                if resolved is not None:
                    isa_snap, orig_uk, orig_ms, src = resolved
                    fallback_meta["ISA"] = (orig_uk, orig_ms, src, orig_err)

            if not invest_snap.ok and invest_transient:
                orig_err = invest_snap.error
                resolved = await self._resolve_account_fallback("Invest", orig_err, db_snapshot_cache)
                if resolved is not None:
                    invest_snap, orig_uk, orig_ms, src = resolved
                    fallback_meta["Invest"] = (orig_uk, orig_ms, src, orig_err)

            isa_configured = bool(self.isa_key and self.isa_secret)
            invest_configured = bool(self.invest_key and self.invest_secret)
            live_count = int(isa_snap.ok and "ISA" not in fallback_meta) + int(
                invest_snap.ok and "Invest" not in fallback_meta
            )

            is_fallback = bool(fallback_meta)
            fallback_source = ""
            fallback_note = ""
            portfolio_uk = now_uk
            portfolio_ms = now_ms

            if fallback_meta:
                sources = {m[2] for m in fallback_meta.values()}
                if live_count == 0:
                    fallback_source = next(iter(sources)) if len(sources) == 1 else "mixed"
                    # Preserve historical snapshot timestamp when no live account succeeded
                    best_meta = max(fallback_meta.values(), key=lambda m: m[1])
                    portfolio_uk = best_meta[0]
                    portfolio_ms = best_meta[1]
                    fallback_note = f"⚠️ T212 实时拉取异常，已回退至 {portfolio_uk} UK 缓存快照 ({fallback_source})"
                else:
                    fallback_source = "mixed"
                    parts = [f"{lbl}@{m[0]} UK" for lbl, m in fallback_meta.items()]
                    fallback_note = f"⚠️ 部分账户实时拉取异常，已回退缓存快照: {', '.join(parts)}"

            cum_map: dict[tuple[str, str], float] = {}
            if self._db is not None:
                if not is_fallback:
                    live_divs = [
                        d.model_dump()
                        for acc in (isa_snap, invest_snap)
                        if acc.ok and not acc.is_fallback
                        for d in acc.recent_dividends
                    ]
                    if live_divs and hasattr(self._db, "upsert_dividends"):
                        try:
                            await self._db.upsert_dividends(live_divs)
                        except Exception as exc:
                            logger.warning("Failed archiving live T212 dividends: %s", exc)
                if hasattr(self._db, "get_cumulative_dividends_by_symbol"):
                    try:
                        cum_map = await self._db.get_cumulative_dividends_by_symbol()
                    except Exception as exc:
                        logger.debug("Failed loading cumulative dividends from DB: %s", exc)

            self._enrich_account_dividends(isa_snap, cum_map)
            self._enrich_account_dividends(invest_snap, cum_map)
            if isa_snap.ok and not isa_snap.is_fallback:
                self._last_good_accounts["ISA"] = (now_uk, now_ms, isa_snap.model_copy(deep=True))
            if invest_snap.ok and not invest_snap.is_fallback:
                self._last_good_accounts["Invest"] = (now_uk, now_ms, invest_snap.model_copy(deep=True))

            # Local GBP+USD wallet split for Invest (API v0 only exposes GBP-converted total cash).
            if self.invest_multi_currency_enabled and invest_snap.ok:
                wallet_records: dict[str, dict[str, Any]] = {}
                if self._db is not None and hasattr(self._db, "get_wallet_balances"):
                    try:
                        wallet_records = await self._db.get_wallet_balances("Invest")
                    except Exception as exc:
                        logger.debug("Failed loading Invest wallet balances: %s", exc)
                if wallet_records or self._db is not None:
                    self._wallet_records = wallet_records
                    invest_snap = reconcile_invest_wallets(
                        invest_snap,
                        wallet_records,
                        self._gbp_usd_rate,
                        mismatch_warn_gbp=self.wallet_mismatch_warn_gbp,
                        reconcile_warn_days=self.wallet_reconcile_warn_days,
                    )
                    invest_snap = invest_snap.model_copy(
                        update={
                            "cash_interest_rate_pct": self.cash_interest_rate_pct,
                            "usd_cash_interest_rate_pct": self.usd_cash_interest_rate_pct,
                        }
                    )

            data = PortfolioData(
                fetched_at_uk=portfolio_uk,
                fetched_at_ms=portfolio_ms,
                isa=isa_snap,
                invest=invest_snap,
                total_combined_value=(
                    (isa_snap.total_account_value if isa_snap.ok else 0.0)
                    + (invest_snap.total_account_value if invest_snap.ok else 0.0)
                ),
                total_combined_cash=(
                    (isa_snap.total_cash if isa_snap.ok else 0.0)
                    + (invest_snap.total_cash if invest_snap.ok else 0.0)
                ),
                cash_interest_rate_pct=self.cash_interest_rate_pct,
                usd_cash_interest_rate_pct=self.usd_cash_interest_rate_pct,
                gbp_usd_rate=self._gbp_usd_rate,
                is_fallback=is_fallback,
                fallback_source=fallback_source,
                fallback_note=fallback_note,
            )

            for lbl, (orig_uk, _, src, orig_err) in fallback_meta.items():
                await self._notify_ops(
                    "WARN",
                    f"t212_fallback:{lbl}",
                    f"T212 {lbl} 账户实时拉取失败（{orig_err[:120]}），已回退至 {src} 快照 @ {orig_uk} UK。",
                )

            if not is_fallback and (isa_snap.ok or invest_snap.ok):
                self._cache = (now, data)
                self._last_good_portfolio = data.model_copy(deep=True)
                if isa_configured and not isa_snap.ok:
                    await self._notify_ops("WARN", "t212_isa_failed", f"T212 ISA 账户拉取失败：{isa_snap.error}")
                if invest_configured and not invest_snap.ok:
                    await self._notify_ops("WARN", "t212_invest_failed", f"T212 Invest 账户拉取失败：{invest_snap.error}")
            elif isa_snap.ok or invest_snap.ok:
                # Partial/full fallback: do not poison TTL cache, still alert if the other configured account failed without fallback
                if isa_configured and not isa_snap.ok:
                    await self._notify_ops("WARN", "t212_isa_failed", f"T212 ISA 账户拉取失败：{isa_snap.error}")
                if invest_configured and not invest_snap.ok:
                    await self._notify_ops("WARN", "t212_invest_failed", f"T212 Invest 账户拉取失败：{invest_snap.error}")
            elif isa_configured or invest_configured:
                await self._notify_ops(
                    "CRITICAL",
                    "t212_all_accounts_failed",
                    f"Trading 212 双账户全部读取失败！ISA: {isa_snap.error} | Invest: {invest_snap.error}",
                    title="T212 双账户全部读取失败",
                    business_blocking=True,
                    bot_message="🚨 Trading 212 双账户全部读取失败，今日持仓与风控不可用，请检查 API 密钥或查看运维告警。",
                )
            return data
