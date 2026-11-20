"""Unified MarketDataService orchestrating bundles, fundamentals cache, US deep intel, and macro snapshots."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yfinance as yf

from src.data_providers.market_data.external_clients import FinnhubClient, FredClient
from src.data_providers.market_data.models_and_parsing import (
    MacroSnapshot,
    TickerMarketBundle,
    _clean_text,
    _compute_safe_fcf_yield_pct,
    _is_us_symbol,
    _parse_date_days_ahead,
    _parse_eps_estimate,
    _pick_peg_ratio,
    _ratio_to_pct,
    _resolve_dividend_yield_pct,
    _safe_float,
    _safe_int,
    apply_basic_financials,
)
from src.data_providers.market_data.yahoo_mixin import YFRateLimitError, YahooHistoryMixin

logger = logging.getLogger("src.data_providers.market_data")


class MarketDataService(YahooHistoryMixin):
    """Async market data aggregator combining yfinance, Finnhub, and FRED with bounded TTL caching."""

    def __init__(
        self,
        finnhub_key: str = "",
        fred_key: str = "",
        ttl_seconds: int = 900,
        max_concurrency: int = 6,
        ops_notifier: Any = None,
        max_cache_entries: int = 512,
        deep_intel_ttl_seconds: int = 21600,
        db: Any = None,
    ) -> None:
        self.finnhub = FinnhubClient(finnhub_key)
        self.fred = FredClient(fred_key)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_cache_entries = max(1, int(max_cache_entries))
        self.deep_intel_ttl_seconds = max(60, int(deep_intel_ttl_seconds))
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ops = ops_notifier
        self._db = db

        self._hist_cache: OrderedDict[tuple[str, str], tuple[float, pd.DataFrame]] = OrderedDict()
        self._bundle_cache: OrderedDict[str, tuple[float, TickerMarketBundle]] = OrderedDict()
        self._deep_intel_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._macro_cache: tuple[float, MacroSnapshot] | None = None

        self._inflight_hist: dict[tuple[str, str], asyncio.Future[pd.DataFrame]] = {}
        self._inflight_bundle: dict[tuple[str, bool], asyncio.Future[tuple[pd.DataFrame, TickerMarketBundle]]] = {}
        self._inflight_macro: asyncio.Future[MacroSnapshot] | None = None

        self._finnhub_http: tuple[Any, httpx.AsyncClient] | None = None
        self._fred_http: tuple[Any, httpx.AsyncClient] | None = None
        self._init_yf_tz_cache()

    def set_ops_notifier(self, ops_notifier: Any) -> None:
        self._ops = ops_notifier

    def set_db(self, db: Any) -> None:
        self._db = db

    async def _notify_ops(self, severity: str, alert_key: str, message: str) -> None:
        if self._ops is not None:
            await self._ops.notify(severity=severity, source="market_data", alert_key=alert_key, message=message)

    @staticmethod
    def _init_yf_tz_cache() -> None:
        """Pre-create yfinance cache directory once to avoid thread race FileExistsError."""
        try:
            cache_dir = Path.home() / ".cache" / "py-yfinance"
            cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_dir))
        except Exception as exc:
            logger.debug("Skipping yfinance tz cache setup: %s", exc)

    async def _get_http_client(self, target: str) -> httpx.AsyncClient:
        """Lazily create/reuse httpx.AsyncClient, rebuilding automatically if test transport changes."""
        if target == "finnhub":
            transport = self.finnhub._transport
            if self._finnhub_http is not None:
                cached_tr, client = self._finnhub_http
                if cached_tr is transport and not client.is_closed:
                    return client
                await client.aclose()
            client = httpx.AsyncClient(transport=transport)
            self._finnhub_http = (transport, client)
            return client

        transport = self.fred._transport
        if self._fred_http is not None:
            cached_tr, client = self._fred_http
            if cached_tr is transport and not client.is_closed:
                return client
            await client.aclose()
        client = httpx.AsyncClient(transport=transport)
        self._fred_http = (transport, client)
        return client

    async def close(self) -> None:
        """Gracefully release internal HTTP client connection pools."""
        for attr in ("_finnhub_http", "_fred_http"):
            pair = getattr(self, attr, None)
            if pair is not None:
                _, client = pair
                if not client.is_closed:
                    await client.aclose()
                setattr(self, attr, None)

    async def get_bundles(
        self,
        symbols: list[str],
        include_news: bool = True,
        include_earnings: bool = True,
        include_info: bool = True,
        isin_map: dict[str, str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, TickerMarketBundle]]:
        now = time.monotonic()
        unique_syms = list(dict.fromkeys(s.strip() for s in symbols if s and s.strip()))
        histories: dict[str, pd.DataFrame] = {}
        bundles: dict[str, TickerMarketBundle] = {}
        failed_syms: list[str] = []

        if not include_info and not include_news and not include_earnings:
            histories = await self.get_histories(unique_syms, period="2y")
            for sym in unique_syms:
                df = histories.get(sym, pd.DataFrame())
                b = TickerMarketBundle(
                    symbol=sym,
                    isin=_clean_text((isin_map or {}).get(sym)).upper(),
                    has_history=not df.empty,
                    current_price=float(df["Close"].iloc[-1]) if not df.empty else None,
                )
                b.identity_summary = b.build_identity_anchor()
                bundles[sym] = b
                if df.empty:
                    failed_syms.append(f"{sym}(empty)")
            await self._emit_batch_failure_ops(unique_syms, failed_syms)
            return histories, bundles

        if self._db is not None and unique_syms:
            missing_h_syms = [
                s
                for s in unique_syms
                if self._get_cached_hist_fallback(s, now) is None
            ]
            if missing_h_syms:
                await self.get_histories(missing_h_syms, period="2y")
                now = time.monotonic()

        loop = asyncio.get_running_loop()

        async def _load_one(sym: str) -> None:
            h_cached = self._get_cached_hist_fallback(sym, now)
            b_cached = self._get_cached_bundle(sym, now, require_info=include_info)
            if h_cached is not None and b_cached is not None:
                histories[sym] = h_cached
                bundles[sym] = self._apply_isin_override(b_cached, isin_map)
                return

            inflight_key = (sym, include_info)
            inflight = self._inflight_bundle.get(inflight_key)
            if inflight is not None:
                try:
                    hist, bundle = await inflight
                    histories[sym] = hist
                    bundles[sym] = self._apply_isin_override(bundle, isin_map)
                    if hist.empty:
                        failed_syms.append(f"{sym}(empty)")
                except Exception as exc:
                    failed_syms.append(f"{sym}({type(exc).__name__})")
                return

            fut: asyncio.Future[tuple[pd.DataFrame, TickerMarketBundle]] = loop.create_future()
            self._inflight_bundle[inflight_key] = fut
            try:
                async with self._sem:
                    hist, bundle = await asyncio.to_thread(
                        self._fetch_yf_sync,
                        sym,
                        "2y",
                        include_info,
                        include_news,
                        include_earnings,
                        h_cached,
                    )
                fetch_now = time.monotonic()
                if not hist.empty:
                    self._set_cached_hist(sym, "2y", fetch_now, hist)
                else:
                    stale_h = self._get_cached_hist(sym, "2y", fetch_now, allow_stale=True)
                    if stale_h is not None:
                        hist = stale_h
                        bundle.has_history = True
                        if bundle.current_price is None:
                            bundle.current_price = float(stale_h["Close"].iloc[-1])
                    failed_syms.append(f"{sym}(empty)")

                if bundle.has_history or bundle.short_name or bundle.current_price is not None:
                    self._set_cached_bundle(sym, fetch_now, bundle)
                else:
                    stale_b = self._get_cached_bundle(sym, fetch_now, require_info=False, allow_stale=True)
                    if stale_b is not None:
                        bundle = stale_b

                histories[sym] = hist
                bundles[sym] = self._apply_isin_override(bundle, isin_map)
                if not fut.done():
                    fut.set_result((hist, bundle))
            except Exception as exc:
                failed_syms.append(f"{sym}({type(exc).__name__})")
                logger.warning("Yahoo Finance fetch failed for %s: %s", sym, exc)
                stale_h = self._get_cached_hist(sym, "2y", time.monotonic(), allow_stale=True)
                stale_b = self._get_cached_bundle(sym, time.monotonic(), require_info=False, allow_stale=True)
                if stale_h is not None:
                    histories[sym] = stale_h
                if stale_b is not None:
                    bundles[sym] = self._apply_isin_override(stale_b, isin_map)
                if not fut.done():
                    if stale_h is not None and stale_b is not None:
                        fut.set_result((stale_h, stale_b))
                    else:
                        fut.set_exception(exc)
            finally:
                self._inflight_bundle.pop(inflight_key, None)

        await asyncio.gather(*(_load_one(s) for s in unique_syms))
        await self._emit_batch_failure_ops(unique_syms, failed_syms)

        # Enrich stocks with Finnhub earnings calendar, news, and ISIN profile fallback
        if self.finnhub.enabled and (include_news or include_earnings or include_info):
            client = await self._get_http_client("finnhub")
            batch_errors: list[str] = []
            if include_earnings:
                cal, cal_err = await self.finnhub._fetch_earnings_calendar(client, days_ahead=21)
                if cal_err:
                    batch_errors.append(cal_err)
                for sym, b in bundles.items():
                    if b.quote_type.upper() == "ETF":
                        continue
                    if sym in cal:
                        ed_str, eps_raw = cal[sym]
                        ed_iso, delta = _parse_date_days_ahead(ed_str)
                        if ed_iso:
                            b.earnings_date = ed_iso
                            b.days_to_earnings = delta
                        eps_val = _parse_eps_estimate(eps_raw)
                        if eps_val is not None:
                            b.eps_estimate = eps_val

            if include_news:
                us_syms = [s for s in unique_syms if _is_us_symbol(s) and s in bundles]
                news_tasks = [self.finnhub._fetch_company_news(client, s, days=7, limit=3) for s in us_syms]
                news_results = await asyncio.gather(*news_tasks, return_exceptions=True)
                for sym, res in zip(us_syms, news_results):
                    if isinstance(res, tuple):
                        items, err = res
                        if err:
                            batch_errors.append(err)
                        if items:
                            bundles[sym].news = items

            if include_info:
                need_profile = [
                    (sym, b)
                    for sym, b in bundles.items()
                    if b.isin
                    and (
                        (not b.short_name or b.short_name.upper() == sym.upper())
                        or (b.quote_type.upper() != "ETF" and not b.sector)
                    )
                ]
                if need_profile:
                    prof_results = await asyncio.gather(
                        *(self.finnhub.get_profile_by_isin(client, b.isin) for _, b in need_profile),
                        return_exceptions=True,
                    )
                    prof_now = time.monotonic()
                    for (sym, b), prof in zip(need_profile, prof_results):
                        if not isinstance(prof, dict) or not prof:
                            continue
                        changed = False
                        prof_name = _clean_text(prof.get("name"))
                        if prof_name and (not b.short_name or b.short_name.upper() == sym.upper()):
                            b.short_name = prof_name
                            changed = True
                        if prof_name and (not b.long_name or b.long_name.upper() == sym.upper()):
                            b.long_name = prof_name
                            changed = True
                        prof_ind = _clean_text(prof.get("industry"))
                        if prof_ind and not b.sector:
                            b.sector = prof_ind
                            changed = True
                        if prof_ind and not b.industry:
                            b.industry = prof_ind
                            changed = True
                        prof_exch = _clean_text(prof.get("exchange"))
                        if prof_exch and not b.exchange:
                            b.exchange = prof_exch
                            changed = True
                        prof_curr = _clean_text(prof.get("currency"))
                        if prof_curr and not b.currency:
                            b.currency = prof_curr
                            changed = True
                        if changed:
                            b.identity_summary = b.build_identity_anchor()
                            self._set_cached_bundle(sym, prof_now, b)

                # IMP-18: PB / EV-EBITDA / current ratio for US equities. Gap-filling only,
                # capped at 40 symbols so a wide /pick shortlist cannot burn the 60/min quota.
                metric_targets = [
                    (sym, b)
                    for sym, b in bundles.items()
                    if _is_us_symbol(sym) and b.quote_type.upper() != "ETF" and b.price_to_book is None
                ][:40]
                if metric_targets:
                    metric_results = await asyncio.gather(
                        *(self.finnhub.get_basic_financials(client, sym) for sym, _ in metric_targets),
                        return_exceptions=True,
                    )
                    metric_now = time.monotonic()
                    enriched = 0
                    for (sym, b), metrics in zip(metric_targets, metric_results):
                        if not isinstance(metrics, dict):
                            continue
                        if apply_basic_financials(b, metrics):
                            self._set_cached_bundle(sym, metric_now, b)
                            enriched += 1
                    if enriched:
                        logger.debug("Finnhub basic financials enriched %d/%d US bundles", enriched, len(metric_targets))

            self.finnhub.last_error = batch_errors[0] if batch_errors else None
            if batch_errors:
                await self._notify_ops(
                    "WARN",
                    "finnhub_error",
                    f"Finnhub 美股新闻/财历接口异常：{batch_errors[0]}",
                )

        return histories, bundles

    @staticmethod
    def _apply_isin_override(
        bundle: TickerMarketBundle,
        isin_map: dict[str, str] | None,
    ) -> TickerMarketBundle:
        if not isin_map:
            return bundle
        override = _clean_text(isin_map.get(bundle.symbol)).upper()
        if not override or override == bundle.isin:
            return bundle
        return bundle.model_copy(
            update={
                "isin": override,
                "identity_summary": bundle.build_identity_anchor(isin_override=override),
            }
        )

    async def _emit_batch_failure_ops(self, unique_syms: list[str], failed_syms: list[str]) -> None:
        if not failed_syms:
            return
        if len(unique_syms) >= 2 and len(failed_syms) >= len(unique_syms):
            await self._notify_ops(
                "CRITICAL",
                "yfinance_all_failed",
                f"Yahoo Finance 行情批量全部拉取失败（{len(failed_syms)}/{len(unique_syms)}）：{', '.join(failed_syms[:5])}",
            )
        else:
            await self._notify_ops(
                "WARN",
                "yfinance_partial_failed",
                f"部分标的行情从 Yahoo Finance 拉取缺失（{len(failed_syms)}/{len(unique_syms)}）：{', '.join(failed_syms[:5])}",
            )

    async def get_us_deep_intel(self, symbol: str) -> dict[str, Any]:
        """On-demand cached US analyst recommendation, EPS surprise & insider sentiment (rate-limit safe)."""
        clean_sym = symbol.strip().upper()
        if not clean_sym or not self.finnhub.enabled or not _is_us_symbol(clean_sym):
            return {}
        now = time.monotonic()
        cached = self._deep_intel_cache.get(clean_sym)
        if cached and (now - cached[0]) < self.deep_intel_ttl_seconds:
            self._deep_intel_cache.move_to_end(clean_sym)
            return cached[1]

        client = await self._get_http_client("finnhub")
        recs, surprises, insider = await asyncio.gather(
            self.finnhub.get_recommendation_trends(client, clean_sym),
            self.finnhub.get_earnings_surprises(client, clean_sym, limit=4),
            self.finnhub.get_insider_sentiment(client, clean_sym, days=90),
        )
        intel = {
            "symbol": clean_sym,
            "recommendations": recs,
            "earnings_surprises": surprises,
            "insider_sentiment": insider,
        }
        if recs or surprises or insider:
            self._deep_intel_cache[clean_sym] = (now, intel)
            self._deep_intel_cache.move_to_end(clean_sym)
            self._prune_ordered_cache(self._deep_intel_cache, now, self.deep_intel_ttl_seconds)
        return intel

    async def get_us_deep_intel_batch(
        self, symbols: list[str], max_concurrency: int = 2
    ) -> dict[str, dict[str, Any]]:
        """Batch-fetch Finnhub US deep intel reusing the 6h cache with bounded concurrency."""
        if not self.finnhub.enabled:
            return {}
        clean_syms = list(
            dict.fromkeys(
                s.strip().upper()
                for s in symbols
                if s and s.strip() and _is_us_symbol(s)
            )
        )
        if not clean_syms:
            return {}

        now = time.monotonic()
        out: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for sym in clean_syms:
            cached = self._deep_intel_cache.get(sym)
            if cached and (now - cached[0]) < self.deep_intel_ttl_seconds:
                self._deep_intel_cache.move_to_end(sym)
                if cached[1]:
                    out[sym] = cached[1]
            else:
                missing.append(sym)

        if missing:
            sem = asyncio.Semaphore(max(1, int(max_concurrency)))

            async def _fetch_one(sym: str) -> tuple[str, dict[str, Any]]:
                async with sem:
                    try:
                        return sym, await self.get_us_deep_intel(sym)
                    except Exception as exc:
                        logger.debug("get_us_deep_intel(%s) failed: %s", sym, exc)
                        return sym, {}

            results = await asyncio.gather(*(_fetch_one(sym) for sym in missing))
            for sym, intel in results:
                if intel:
                    out[sym] = intel
        return out

    def _fetch_single_info_sync(self, symbol: str) -> dict[str, Any]:
        """Fetch raw fundamental fields from yfinance.Ticker(symbol).info with safe retries."""
        def _load_info() -> dict[str, Any]:
            data = yf.Ticker(symbol).info or {}
            return data if isinstance(data, dict) else {}

        info = self._run_yf_call_with_retry(_load_info, max_retries=1, base_delay=2.0) or {}
        if not info:
            return {}
        curr = _clean_text(info.get("currency"))
        cur_price = (
            _safe_float(info.get("currentPrice"))
            or _safe_float(info.get("regularMarketPrice"))
            or _safe_float(info.get("previousClose"))
        )
        return {
            "symbol": symbol.strip().upper(),
            "short_name": _clean_text(info.get("shortName")) or _clean_text(info.get("longName")) or symbol,
            "exchange": _clean_text(info.get("exchange")) or _clean_text(info.get("fullExchangeName")),
            "currency": curr,
            "sector": _clean_text(info.get("sector")),
            "industry": _clean_text(info.get("industry")),
            "quote_type": _clean_text(info.get("quoteType")),
            "market_cap": _safe_float(info.get("marketCap")),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "roe_pct": _ratio_to_pct(info.get("returnOnEquity")),
            "profit_margin_pct": _ratio_to_pct(info.get("profitMargins")),
            "revenue_growth_pct": _ratio_to_pct(info.get("revenueGrowth")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "dividend_yield_pct": _resolve_dividend_yield_pct(info, cur_price, curr),
            "analyst_count": _safe_int(info.get("numberOfAnalystOpinions")),
            "recommendation_key": _clean_text(info.get("recommendationKey")),
            "beta": _safe_float(info.get("beta")),
            "target_mean_price": _safe_float(info.get("targetMeanPrice")),
            "target_high_price": _safe_float(info.get("targetHighPrice")),
            "target_low_price": _safe_float(info.get("targetLowPrice")),
            "peg_ratio": _pick_peg_ratio(info),
            "fcf_yield_pct": _compute_safe_fcf_yield_pct(info.get("freeCashflow"), info.get("marketCap")),
            "updated_at_ms": int(time.time() * 1000),
        }

    async def get_fundamentals_cached(
        self,
        symbols: list[str],
        *,
        max_age_days: int = 21,
        concurrency: int = 2,
        delay_seconds: float = 0.45,
        max_consecutive_429: int = 3,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Load cached fundamentals from SQLite and fetch stale/missing ones with rate-limit circuit breaker."""
        clean_syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
        results: dict[str, dict[str, Any]] = {}
        if not clean_syms:
            return results, {"cached": 0, "fetched": 0, "failed": 0, "skipped": 0, "rate_limited": False}

        if self._db is not None and hasattr(self._db, "get_fundamentals_cache"):
            try:
                results.update(await self._db.get_fundamentals_cache(clean_syms, max_age_days=max_age_days))
            except Exception as exc:
                logger.warning("Failed reading fundamentals_cache: %s", exc)

        cached_count = len(results)
        missing = [s for s in clean_syms if s not in results]
        if not missing:
            return results, {
                "cached": cached_count,
                "fetched": 0,
                "failed": 0,
                "skipped": 0,
                "rate_limited": False,
            }

        sem = asyncio.Semaphore(max(1, min(int(concurrency), 4)))
        pending_flush: list[dict[str, Any]] = []
        state = {"fetched": 0, "failed": 0, "skipped": 0, "consec_429": 0, "rate_limited": False}
        state_lock = asyncio.Lock()

        async def _flush_pending(force: bool = False) -> None:
            if self._db is None or not hasattr(self._db, "upsert_fundamentals_cache"):
                pending_flush.clear()
                return
            if not pending_flush or (not force and len(pending_flush) < 20):
                return
            batch = list(pending_flush)
            pending_flush.clear()
            try:
                await self._db.upsert_fundamentals_cache(batch)
            except Exception as exc:
                logger.warning("Checkpoint fundamentals_cache flush failed: %s", exc)

        async def _worker(sym: str) -> None:
            async with sem:
                async with state_lock:
                    if state["rate_limited"]:
                        state["skipped"] += 1
                        return
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                try:
                    info = await asyncio.to_thread(self._fetch_single_info_sync, sym)
                    async with state_lock:
                        state["consec_429"] = 0
                        if info:
                            results[sym] = info
                            pending_flush.append(info)
                            state["fetched"] += 1
                            await _flush_pending(force=False)
                        else:
                            state["failed"] += 1
                except Exception as exc:
                    msg = str(exc).lower()
                    is_429 = isinstance(exc, YFRateLimitError) or "429" in msg or "too many requests" in msg
                    async with state_lock:
                        state["failed"] += 1
                        if is_429:
                            state["consec_429"] += 1
                            if state["consec_429"] >= max(1, int(max_consecutive_429)):
                                state["rate_limited"] = True
                                logger.warning(
                                    "Yahoo Finance .info circuit breaker tripped after %d consecutive 429s; "
                                    "aborting remaining fetches safely.",
                                    state["consec_429"],
                                )
                        else:
                            state["consec_429"] = 0

        await asyncio.gather(*(_worker(sym) for sym in missing))
        await _flush_pending(force=True)
        return results, {
            "cached": cached_count,
            "fetched": state["fetched"],
            "failed": state["failed"],
            "skipped": state["skipped"],
            "rate_limited": state["rate_limited"],
        }

    async def get_macro_snapshot(self) -> MacroSnapshot:
        now = time.monotonic()
        if self._macro_cache and self.ttl_seconds > 0 and (now - self._macro_cache[0]) < self.ttl_seconds:
            return self._macro_cache[1]

        if self._inflight_macro is not None:
            return await self._inflight_macro

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[MacroSnapshot] = loop.create_future()
        self._inflight_macro = fut

        try:
            def _ret_5d(df: pd.DataFrame) -> float | None:
                if df.empty or "Close" not in df.columns:
                    return None
                closes = df["Close"].dropna()
                if len(closes) < 6:
                    return None
                prev = float(closes.iloc[-6])
                if prev <= 0:
                    return None
                return ((float(closes.iloc[-1]) / prev) - 1.0) * 100.0

            def _ret_1d(df: pd.DataFrame) -> float | None:
                # v5.5.8: same-day move from the last two cached closes (zero extra requests).
                if df.empty or "Close" not in df.columns:
                    return None
                closes = df["Close"].dropna()
                if len(closes) < 2:
                    return None
                prev = float(closes.iloc[-2])
                if prev <= 0:
                    return None
                return ((float(closes.iloc[-1]) / prev) - 1.0) * 100.0

            def _last_close(df: pd.DataFrame) -> float | None:
                if df.empty or "Close" not in df.columns:
                    return None
                closes = df["Close"].dropna()
                return float(closes.iloc[-1]) if not closes.empty else None

            fx_syms = ["GBPUSD=X", "EURGBP=X", "SPY", "QQQ", "ISF.L", "^VIX"]
            h_map: dict[str, pd.DataFrame] = {}

            async def _pull_hist(sym: str) -> None:
                try:
                    async with self._sem:
                        df, _ = await asyncio.to_thread(
                            self._fetch_yf_sync, sym, "1mo", False, False, False
                        )
                        h_map[sym] = df
                except Exception as exc:
                    logger.warning("Macro history fetch failed for %s: %s", sym, exc)

            async def _pull_fred() -> dict[str, dict[str, float | None]]:
                if not self.fred.enabled:
                    return {}
                client = await self._get_http_client("fred")
                series_ids = [
                    "DGS10",
                    "DFF",
                    "UNRATE",
                    "VIXCLS",
                    "DGS2",
                    "T10Y2Y",
                    "T10YIE",
                    "ECBDFR",
                    "BAMLH0A0HYM2",
                ]
                pairs = await asyncio.gather(
                    *(self.fred._fetch_series_stats(client, sid, limit=25) for sid in series_ids)
                )
                out: dict[str, dict[str, float | None]] = {}
                errs: list[str] = []
                for sid, (stats, err) in zip(series_ids, pairs):
                    out[sid] = stats
                    if err:
                        errs.append(err)
                self.fred.last_error = errs[0] if errs else None
                if (
                    errs
                    and out.get("DGS10", {}).get("latest") is None
                    and out.get("DFF", {}).get("latest") is None
                    and out.get("UNRATE", {}).get("latest") is None
                ):
                    await self._notify_ops(
                        "WARN",
                        "fred_error",
                        f"FRED 宏观利率接口异常（{errs[0]}），美债收益率/Fed利率暂缺。",
                    )
                return out

            _, fred_stats = await asyncio.gather(
                asyncio.gather(*(_pull_hist(s) for s in fx_syms)),
                _pull_fred(),
            )

            dgs10_stats = fred_stats.get("DGS10", {})
            us10y = dgs10_stats.get("latest")
            dgs10_chg20 = dgs10_stats.get("change_20d")
            us2y = fred_stats.get("DGS2", {}).get("latest")
            spread_10_2 = fred_stats.get("T10Y2Y", {}).get("latest")
            if spread_10_2 is None and us10y is not None and us2y is not None:
                spread_10_2 = round(us10y - us2y, 4)
            hy_oas_stats = fred_stats.get("BAMLH0A0HYM2", {})
            hy_oas_chg20 = hy_oas_stats.get("change_20d")

            snap = MacroSnapshot(
                fetched_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                gbp_usd=_last_close(h_map.get("GBPUSD=X", pd.DataFrame())),
                eur_gbp=_last_close(h_map.get("EURGBP=X", pd.DataFrame())),
                us_10y_yield=us10y,
                us_10y_change_20d_bps=round(dgs10_chg20 * 100.0, 1) if dgs10_chg20 is not None else None,
                us_2y_yield=us2y,
                yield_spread_10y_2y=spread_10_2,
                breakeven_10y_pct=fred_stats.get("T10YIE", {}).get("latest"),
                hy_oas_spread_pct=hy_oas_stats.get("latest"),
                hy_oas_change_20d_bps=round(hy_oas_chg20 * 100.0, 1) if hy_oas_chg20 is not None else None,
                fed_funds_rate=fred_stats.get("DFF", {}).get("latest"),
                ecb_deposit_rate=fred_stats.get("ECBDFR", {}).get("latest"),
                vix=_last_close(h_map.get("^VIX", pd.DataFrame())) or fred_stats.get("VIXCLS", {}).get("latest"),
                us_unemployment_pct=fred_stats.get("UNRATE", {}).get("latest"),
                spy_5d_pct=_ret_5d(h_map.get("SPY", pd.DataFrame())),
                qqq_5d_pct=_ret_5d(h_map.get("QQQ", pd.DataFrame())),
                ftse_5d_pct=_ret_5d(h_map.get("ISF.L", pd.DataFrame())),
                spy_1d_pct=_ret_1d(h_map.get("SPY", pd.DataFrame())),
                qqq_1d_pct=_ret_1d(h_map.get("QQQ", pd.DataFrame())),
                ftse_1d_pct=_ret_1d(h_map.get("ISF.L", pd.DataFrame())),
            )
            if snap.gbp_usd is None and snap.vix is None and snap.spy_5d_pct is None:
                await self._notify_ops(
                    "CRITICAL",
                    "macro_data_unavailable",
                    "宏观与外汇（GBPUSD / VIX / SPY）核心行情全部获取失败，请检查网络或 Yahoo Finance 连通性。",
                )
                if self._macro_cache is not None:
                    stale_snap = self._macro_cache[1].model_copy(update={"is_stale": True})
                    if not fut.done():
                        fut.set_result(stale_snap)
                    return stale_snap
                if not fut.done():
                    fut.set_result(snap)
                return snap

            self._macro_cache = (time.monotonic(), snap)
            if not fut.done():
                fut.set_result(snap)
            return snap
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._inflight_macro = None
