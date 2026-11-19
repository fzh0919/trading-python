"""Yahoo Finance history/prepost fetching, retry, and TTL memory-cache mixin for MarketDataService."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:  # pragma: no cover - defensive import across yfinance versions
    class YFRateLimitError(Exception):
        pass

from src.data_providers.market_data.models_and_parsing import (
    TickerMarketBundle,
    _clean_text,
    _compute_safe_fcf_yield_pct,
    _deserialize_ohlcv_json,
    _fetch_yahoo_chart_sync,
    _is_us_symbol,
    _parse_date_days_ahead,
    _pick_peg_ratio,
    _ratio_to_pct,
    _resolve_dividend_yield_pct,
    _safe_float,
    _safe_int,
    _sanitize_history_df,
    _sanitize_intraday_df,
    _serialize_ohlcv_df,
)

logger = logging.getLogger("src.data_providers.market_data")


class YahooHistoryMixin:
    def _prune_ordered_cache(self, cache: OrderedDict[Any, tuple[float, Any]], now: float, ttl: int) -> None:
        expired_keys = [k for k, (ts, _) in cache.items() if ttl <= 0 or (now - ts) >= ttl]
        for k in expired_keys:
            cache.pop(k, None)
        while len(cache) > self.max_cache_entries:
            cache.popitem(last=False)

    def _get_cached_hist(self, symbol: str, period: str, now: float, *, allow_stale: bool = False) -> pd.DataFrame | None:
        candidate_keys = [(symbol, period)]
        if period == "1y":
            candidate_keys.append((symbol, "2y"))
        elif allow_stale and period == "2y":
            candidate_keys.append((symbol, "1y"))
        for key in candidate_keys:
            entry = self._hist_cache.get(key)
            if entry is None:
                continue
            ts, df = entry
            if df.empty:
                continue
            if allow_stale or (self.ttl_seconds > 0 and (now - ts) < self.ttl_seconds):
                self._hist_cache.move_to_end(key)
                return df
        return None

    def _get_cached_hist_fallback(
        self,
        symbol: str,
        now: float,
        periods: tuple[str, ...] = ("2y", "1y"),
        *,
        allow_stale: bool = False,
    ) -> pd.DataFrame | None:
        """Return the first cached history DataFrame across periods without ambiguous DataFrame truthiness."""
        for period in periods:
            cached = self._get_cached_hist(symbol, period, now, allow_stale=allow_stale)
            if cached is not None:
                return cached
        return None

    def _set_cached_hist(self, symbol: str, period: str, now: float, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        key = (symbol, period)
        self._hist_cache[key] = (now, df)
        self._hist_cache.move_to_end(key)
        # Keep stale entries up to 2x TTL for error fallback unless capacity exceeded
        self._prune_ordered_cache(self._hist_cache, now, max(self.ttl_seconds * 4, 3600))

    def _get_cached_bundle(
        self,
        symbol: str,
        now: float,
        *,
        require_info: bool = True,
        allow_stale: bool = False,
    ) -> TickerMarketBundle | None:
        entry = self._bundle_cache.get(symbol)
        if entry is None:
            return None
        ts, bundle = entry
        if require_info and not bundle.short_name:
            return None
        if allow_stale or (self.ttl_seconds > 0 and (now - ts) < self.ttl_seconds):
            self._bundle_cache.move_to_end(symbol)
            return bundle
        return None

    def _set_cached_bundle(self, symbol: str, now: float, bundle: TickerMarketBundle) -> None:
        if not bundle.has_history and not bundle.short_name and bundle.current_price is None:
            return
        self._bundle_cache[symbol] = (now, bundle)
        self._bundle_cache.move_to_end(symbol)
        self._prune_ordered_cache(self._bundle_cache, now, max(self.ttl_seconds * 4, 3600))

    @staticmethod
    def _reset_yfinance_crumb_cache() -> None:
        """Best-effort flush of yfinance's cached crumb/cookie after HTTP 401 Invalid Crumb.

        Defensive by design: different yfinance versions expose different internals and offline
        mocked environments may not provide `yf.data` at all, so every probe is wrapped.
        """
        try:
            yf_data_mod = getattr(yf, "data", None)
            yf_data_cls = getattr(yf_data_mod, "YfData", None) if yf_data_mod is not None else None
            if yf_data_cls is not None:
                try:
                    yf_data_obj: Any = yf_data_cls()
                except Exception:
                    yf_data_obj = None
                if yf_data_obj is not None:
                    for attr in ("_crumb", "_cookie", "_cookie_str", "_cookies"):
                        try:
                            setattr(yf_data_obj, attr, None)
                        except Exception:
                            continue
        except Exception:
            pass
        try:
            yf_cache = getattr(yf, "cache", None)
            clear_fn = getattr(yf_cache, "clear", None)
            if callable(clear_fn):
                clear_fn()
        except Exception:
            pass

    @staticmethod
    def _run_yf_call_with_retry(fn: Any, *, max_retries: int = 2, base_delay: float = 0.8) -> Any:
        """Retry transient yfinance rate limits or temporary failures with fresh Ticker instantiation."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except YFRateLimitError as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(max(base_delay, 2.0) * (2 ** attempt))
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                crumb_error = any(tok in msg for tok in ("401", "invalid crumb", "unauthorized"))
                if crumb_error:
                    if attempt < max_retries:
                        YahooHistoryMixin._reset_yfinance_crumb_cache()
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    raise
                transient = any(tok in msg for tok in ("429", "too many requests", "timed out", "timeout", "temporarily"))
                if transient and attempt < max_retries:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        return None

    def _fetch_yf_sync(
        self,
        symbol: str,
        period: str = "1y",
        include_info: bool = True,
        include_news: bool = True,
        include_earnings: bool = True,
        preloaded_hist: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, TickerMarketBundle]:
        if preloaded_hist is not None and not preloaded_hist.empty:
            hist = _sanitize_history_df(preloaded_hist)
        else:
            def _load_hist() -> pd.DataFrame | None:
                return yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)

            hist = _sanitize_history_df(self._run_yf_call_with_retry(_load_hist))

        bundle = TickerMarketBundle(symbol=symbol, has_history=not hist.empty)
        if not hist.empty and "Close" in hist.columns:
            bundle.current_price = float(hist["Close"].iloc[-1])

        if not include_info:
            return hist, bundle

        info: dict[str, Any] = {}
        try:
            def _load_info() -> dict[str, Any]:
                t = yf.Ticker(symbol)
                data = t.info or {}
                return data if isinstance(data, dict) else {}

            info = self._run_yf_call_with_retry(_load_info) or {}
            bundle.quote_type = _clean_text(info.get("quoteType"))
            bundle.currency = _clean_text(info.get("currency"))
            bundle.short_name = _clean_text(info.get("shortName")) or _clean_text(info.get("longName")) or symbol
            bundle.long_name = _clean_text(info.get("longName")) or bundle.short_name
            bundle.exchange = _clean_text(info.get("exchange")) or _clean_text(info.get("fullExchangeName"))
            bundle.sector = _clean_text(info.get("sector"))
            bundle.industry = _clean_text(info.get("industry"))
            bundle.isin = _clean_text(info.get("isin")).upper()

            if bundle.current_price is None:
                bundle.current_price = (
                    _safe_float(info.get("currentPrice"))
                    or _safe_float(info.get("regularMarketPrice"))
                    or _safe_float(info.get("previousClose"))
                )
            bundle.trailing_pe = _safe_float(info.get("trailingPE"))
            bundle.forward_pe = _safe_float(info.get("forwardPE"))
            bundle.market_cap = _safe_float(info.get("marketCap"))
            bundle.dividend_yield_pct = _resolve_dividend_yield_pct(info, bundle.current_price, bundle.currency)
            bundle.target_mean_price = _safe_float(info.get("targetMeanPrice"))
            bundle.beta = _safe_float(info.get("beta"))
            bundle.profit_margin_pct = _ratio_to_pct(info.get("profitMargins"))
            bundle.revenue_growth_pct = _ratio_to_pct(info.get("revenueGrowth"))
            bundle.roe_pct = _ratio_to_pct(info.get("returnOnEquity"))
            bundle.debt_to_equity = _safe_float(info.get("debtToEquity"))
            bundle.analyst_count = _safe_int(info.get("numberOfAnalystOpinions"))
            bundle.recommendation_key = _clean_text(info.get("recommendationKey"))
            # v5.5.1 dual-philosophy extras
            bundle.target_high_price = _safe_float(info.get("targetHighPrice"))
            bundle.target_low_price = _safe_float(info.get("targetLowPrice"))
            bundle.peg_ratio = _pick_peg_ratio(info)
            bundle.fcf_yield_pct = _compute_safe_fcf_yield_pct(
                info.get("freeCashflow"),
                info.get("marketCap"),
            )

            ex_div_str, ex_div_delta = _parse_date_days_ahead(info.get("exDividendDate"))
            if ex_div_str and (ex_div_delta is None or ex_div_delta >= -3):
                bundle.ex_dividend_date = ex_div_str
                bundle.days_to_ex_dividend = ex_div_delta
        except Exception as exc:
            logger.debug("yfinance info failed for %s: %s", symbol, exc)

        # Fallback earnings & ex-dividend date from yfinance calendar
        if include_earnings and bundle.quote_type.upper() != "ETF":
            try:
                cal = yf.Ticker(symbol).calendar
                if isinstance(cal, dict):
                    if cal.get("Earnings Date"):
                        ed_list = cal["Earnings Date"]
                        first_ed = ed_list[0] if isinstance(ed_list, (list, tuple)) and ed_list else ed_list
                        ed_str, delta = _parse_date_days_ahead(first_ed)
                        if ed_str and (delta is None or delta >= -2):
                            bundle.earnings_date = ed_str
                            bundle.days_to_earnings = delta
                    if bundle.ex_dividend_date is None and cal.get("Ex-Dividend Date"):
                        ex_str, ex_delta = _parse_date_days_ahead(cal["Ex-Dividend Date"])
                        if ex_str and (ex_delta is None or ex_delta >= -3):
                            bundle.ex_dividend_date = ex_str
                            bundle.days_to_ex_dividend = ex_delta
            except Exception:
                pass

        # Fallback news from yfinance (helpful for UK/EU stocks & ETFs)
        if include_news:
            try:
                raw_news = yf.Ticker(symbol).news or []
                parsed_news: list[dict[str, str]] = []
                for item in raw_news[:3]:
                    content = item.get("content") if isinstance(item.get("content"), dict) else item
                    title = str(content.get("title") or "").strip()
                    if not title:
                        continue
                    pub = str(content.get("pubDate") or "")[:10]
                    provider = ""
                    if isinstance(content.get("provider"), dict):
                        provider = str(content["provider"].get("displayName") or "")
                    parsed_news.append({"date": pub, "source": provider or "Yahoo", "title": title})
                bundle.news = parsed_news
            except Exception:
                pass

        bundle.identity_summary = bundle.build_identity_anchor()
        return hist, bundle

    @staticmethod
    def _extract_batch_frames(
        raw: pd.DataFrame | None,
        batch_syms: list[str],
        *,
        intraday: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Split yf.download output (MultiIndex or single-symbol flat DataFrame) per ticker."""
        out: dict[str, pd.DataFrame] = {}
        if raw is None or raw.empty:
            return out
        sanitizer = _sanitize_intraday_df if intraday else _sanitize_history_df

        if not isinstance(raw.columns, pd.MultiIndex):
            if len(batch_syms) == 1:
                cleaned = sanitizer(raw)
                if not cleaned.empty:
                    out[batch_syms[0]] = cleaned
            return out

        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        for sym in batch_syms:
            sub: pd.DataFrame | None = None
            if sym in lvl0:
                sub = raw[sym]
            elif sym in lvl1:
                sub = raw.xs(sym, axis=1, level=1)
            if sub is not None and not sub.empty:
                cleaned = sanitizer(sub)
                if not cleaned.empty:
                    out[sym] = cleaned
        return out

    def _download_batch_sync(self, batch_syms: list[str], period: str) -> dict[str, pd.DataFrame]:
        if not batch_syms:
            return {}
        try:
            def _do_download() -> pd.DataFrame:
                return yf.download(
                    tickers=batch_syms if len(batch_syms) > 1 else batch_syms[0],
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )

            raw = self._run_yf_call_with_retry(_do_download, max_retries=1)
            return self._extract_batch_frames(raw, batch_syms)
        except Exception as exc:
            logger.debug("yfinance batch download failed (%d syms): %s", len(batch_syms), exc)
            return {}

    async def _download_histories_raw(self, symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
        """Download daily OHLCV for symbols in chunks of 30 with single-ticker fallback."""
        batch_map: dict[str, pd.DataFrame] = {}
        chunk_size = 30
        for idx in range(0, len(symbols), chunk_size):
            if idx > 0 and len(symbols) > 60:
                await asyncio.sleep(0.35)
            chunk = symbols[idx : idx + chunk_size]
            async with self._sem:
                chunk_res = await asyncio.to_thread(self._download_batch_sync, chunk, period)
            batch_map.update(chunk_res)

        missing = [s for s in symbols if s not in batch_map or batch_map[s].empty]
        transport_failures: set[str] = set()
        if missing:
            async def _fallback_one(sym: str) -> tuple[str, pd.DataFrame]:
                try:
                    async with self._sem:
                        df, _ = await asyncio.to_thread(
                            self._fetch_yf_sync,
                            sym,
                            period,
                            False,
                            False,
                            False,
                        )
                    return sym, df
                except Exception as exc:
                    # Only transport-level failures justify the extra HTTP transport below;
                    # a symbol that cleanly returns empty is genuinely absent (delisted/typo).
                    transport_failures.add(sym)
                    logger.debug("Fallback history fetch failed for %s: %s", sym, exc)
                    return sym, pd.DataFrame()

            fb_pairs = await asyncio.gather(*(_fallback_one(s) for s in missing))
            for sym, df in fb_pairs:
                if not df.empty:
                    batch_map[sym] = df

        # IMP-17: last-resort transport, only for symbols whose yfinance call actually raised
        # (429 / crumb / session failures). Cleanly-empty symbols are never re-fetched.
        still_missing = [s for s in symbols if s in transport_failures]
        if still_missing:
            async def _chart_one(sym: str) -> tuple[str, pd.DataFrame]:
                try:
                    async with self._sem:
                        frame = await asyncio.to_thread(_fetch_yahoo_chart_sync, sym, period)
                    return sym, frame
                except Exception as exc:
                    logger.debug("Yahoo chart fallback failed for %s: %s", sym, exc)
                    return sym, pd.DataFrame()

            chart_pairs = await asyncio.gather(*(_chart_one(s) for s in still_missing))
            for sym, frame in chart_pairs:
                if not frame.empty:
                    batch_map[sym] = frame
        return batch_map

    async def get_histories(self, symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
        """Batch-fetch daily OHLCV DataFrames with SQLite incremental merge, split detection & singleflight."""
        now = time.monotonic()
        unique_syms = list(dict.fromkeys(s.strip() for s in symbols if s and s.strip()))
        results: dict[str, pd.DataFrame] = {}
        wait_futures: dict[str, asyncio.Future[pd.DataFrame]] = {}
        to_fetch: list[str] = []
        owned_futures: dict[str, asyncio.Future[pd.DataFrame]] = {}

        loop = asyncio.get_running_loop()
        for sym in unique_syms:
            cached = self._get_cached_hist(sym, period, now)
            if cached is not None:
                results[sym] = cached
                continue
            inflight = self._inflight_hist.get((sym, period))
            if inflight is not None:
                wait_futures[sym] = inflight
                continue
            fut: asyncio.Future[pd.DataFrame] = loop.create_future()
            self._inflight_hist[(sym, period)] = fut
            owned_futures[sym] = fut
            to_fetch.append(sym)

        if to_fetch:
            try:
                batch_map: dict[str, pd.DataFrame] = {}
                local_df_map: dict[str, pd.DataFrame] = {}
                db_period_map: dict[str, str] = {}
                inc_fetch_syms: list[str] = []
                full_fetch_syms: list[str] = []
                to_persist: dict[str, tuple[str, pd.DataFrame]] = {}

                if self._db is not None and hasattr(self._db, "get_cached_ohlcv") and period in ("1y", "2y"):
                    db_cache: dict[str, dict[str, Any]] = {}
                    try:
                        db_cache = await self._db.get_cached_ohlcv(to_fetch, period=period)
                        if period == "1y":
                            miss_db = [s for s in to_fetch if s not in db_cache]
                            if miss_db:
                                db_cache.update(await self._db.get_cached_ohlcv(miss_db, period="2y"))
                    except Exception as exc:
                        logger.debug("SQLite ohlcv_cache lookup failed: %s", exc)
                        db_cache = {}

                    today_utc = datetime.now(timezone.utc).date()
                    for sym in to_fetch:
                        row = db_cache.get(sym)
                        if row:
                            local_df = _deserialize_ohlcv_json(str(row.get("payload_json") or ""))
                            if not local_df.empty:
                                local_df_map[sym] = local_df
                                db_period_map[sym] = str(row.get("period") or period)
                                last_bar_date = pd.Timestamp(local_df.index[-1]).date()
                                if (today_utc - last_bar_date).days <= 5:
                                    inc_fetch_syms.append(sym)
                                    continue
                        full_fetch_syms.append(sym)
                else:
                    full_fetch_syms = list(to_fetch)

                if inc_fetch_syms:
                    inc_map = await self._download_histories_raw(inc_fetch_syms, "1mo")
                    for sym in inc_fetch_syms:
                        local_df = local_df_map[sym]
                        inc_df = inc_map.get(sym, pd.DataFrame())
                        if inc_df.empty:
                            batch_map[sym] = local_df
                            continue
                        overlap_idx = local_df.index.intersection(inc_df.index)
                        if len(overlap_idx) == 0:
                            full_fetch_syms.append(sym)
                            continue
                        max_rel_diff = 0.0
                        for ts in overlap_idx:
                            c_old = float(local_df.loc[ts, "Close"])
                            c_new = float(inc_df.loc[ts, "Close"])
                            if c_old > 0:
                                rel_diff = abs(c_new - c_old) / c_old
                                if rel_diff > max_rel_diff:
                                    max_rel_diff = rel_diff
                        if max_rel_diff <= 0.005:
                            merged_df = _sanitize_history_df(pd.concat([local_df, inc_df]))
                            if not merged_df.empty:
                                batch_map[sym] = merged_df
                                to_persist[sym] = (db_period_map.get(sym, period), merged_df)
                            else:
                                full_fetch_syms.append(sym)
                        else:
                            logger.info(
                                "Split/corporate adjustment detected for %s (overlap diff=%.2f%% > 0.5%%); refetching full %s.",
                                sym,
                                max_rel_diff * 100.0,
                                period,
                            )
                            full_fetch_syms.append(sym)

                full_fetch_syms = list(dict.fromkeys(full_fetch_syms))
                if full_fetch_syms:
                    full_map = await self._download_histories_raw(full_fetch_syms, period)
                    for sym in full_fetch_syms:
                        df = full_map.get(sym, pd.DataFrame())
                        if not df.empty:
                            batch_map[sym] = df
                            to_persist[sym] = (period, df)
                        elif sym in local_df_map and not local_df_map[sym].empty:
                            batch_map[sym] = local_df_map[sym]

                if to_persist and self._db is not None and hasattr(self._db, "save_cached_ohlcv") and period in ("1y", "2y"):
                    entries = [
                        ent
                        for sym, (save_p, df) in to_persist.items()
                        if (ent := _serialize_ohlcv_df(sym, save_p, df)) is not None
                    ]
                    if entries:
                        try:
                            await self._db.save_cached_ohlcv(entries)
                        except Exception as exc:
                            logger.warning("Failed persisting ohlcv_cache entries: %s", exc)

                fetch_now = time.monotonic()
                for sym in to_fetch:
                    df = batch_map.get(sym, pd.DataFrame())
                    if not df.empty:
                        if sym in to_persist or self._db is None:
                            self._set_cached_hist(sym, period, fetch_now, df)
                        results[sym] = df
                    else:
                        stale = self._get_cached_hist(sym, period, fetch_now, allow_stale=True)
                        df = stale if stale is not None else pd.DataFrame()
                        results[sym] = df
                    fut = owned_futures[sym]
                    if not fut.done():
                        fut.set_result(df)
            except Exception as exc:
                for sym, fut in owned_futures.items():
                    if not fut.done():
                        fut.set_exception(exc)
                raise
            finally:
                for sym in to_fetch:
                    self._inflight_hist.pop((sym, period), None)

        if wait_futures:
            waited = await asyncio.gather(*wait_futures.values(), return_exceptions=True)
            for sym, val in zip(wait_futures.keys(), waited):
                if isinstance(val, pd.DataFrame):
                    results[sym] = val
                else:
                    results[sym] = pd.DataFrame()

        return results

    async def get_history(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        clean_sym = symbol.strip()
        if not clean_sym:
            return pd.DataFrame()
        hist_map = await self.get_histories([clean_sym], period=period)
        return hist_map.get(clean_sym, pd.DataFrame())

    async def get_us_realtime_quote(self, symbol: str) -> dict[str, Any] | None:
        """On-demand fast US real-time quote via Finnhub /quote (returns None if disabled or non-US)."""
        if not self.finnhub.enabled or not _is_us_symbol(symbol):
            return None
        client = await self._get_http_client("finnhub")
        return await self.finnhub.get_quote(client, symbol)

    def _download_prepost_15m_sync(self, batch_syms: list[str]) -> dict[str, pd.DataFrame]:
        if not batch_syms:
            return {}
        try:
            def _do_download() -> pd.DataFrame:
                return yf.download(
                    tickers=batch_syms if len(batch_syms) > 1 else batch_syms[0],
                    period="5d",
                    interval="15m",
                    prepost=True,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )

            raw = self._run_yf_call_with_retry(_do_download, max_retries=1)
            return self._extract_batch_frames(raw, batch_syms, intraday=True)
        except Exception as exc:
            logger.debug("yfinance prepost 15m batch download failed (%d syms): %s", len(batch_syms), exc)
            return {}

    def _fetch_prepost_15m_single_sync(self, symbol: str) -> pd.DataFrame:
        try:
            def _load() -> pd.DataFrame | None:
                return yf.Ticker(symbol).history(
                    period="5d",
                    interval="15m",
                    prepost=True,
                    auto_adjust=True,
                )

            raw = self._run_yf_call_with_retry(_load, max_retries=1)
            return _sanitize_intraday_df(raw)
        except Exception as exc:
            logger.debug("yfinance prepost 15m single fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def _extract_intraday_15m_quote(self, sym: str, df: pd.DataFrame | None) -> dict[str, Any] | None:
        clean = _sanitize_intraday_df(df)
        if clean.empty or "Close" not in clean.columns:
            return None
        closes = clean["Close"].astype(float)
        idx = pd.DatetimeIndex(closes.index)
        et_idx = idx.tz_convert(ZoneInfo("America/New_York")) if idx.tz is not None else idx
        current_price = float(closes.iloc[-1])
        if not math.isfinite(current_price) or current_price <= 0:
            return None

        dates = [ts.date() for ts in et_idx]
        unique_dates = sorted(set(dates))
        prev_close: float | None = None
        if len(unique_dates) >= 2:
            prev_date = unique_dates[-2]
            prev_mask = [d == prev_date for d in dates]
            prev_bars = closes[prev_mask]
            prev_et = [ts for ts, m in zip(et_idx, prev_mask) if m]
            reg_bars = [
                float(val)
                for val, ts in zip(prev_bars, prev_et)
                if 570 <= (ts.hour * 60 + ts.minute) <= 960
            ]
            if reg_bars:
                prev_close = reg_bars[-1]
            elif not prev_bars.empty:
                prev_close = float(prev_bars.iloc[-1])
        else:
            daily_df = self._get_cached_hist_fallback(sym, time.monotonic(), allow_stale=True)
            if daily_df is not None and not daily_df.empty and "Close" in daily_df.columns:
                d_closes = pd.to_numeric(daily_df["Close"], errors="coerce").dropna()
                if not d_closes.empty:
                    d_dates = [pd.Timestamp(ts).date() for ts in d_closes.index]
                    prior = [float(v) for v, d in zip(d_closes, d_dates) if d < unique_dates[0]]
                    if prior:
                        prev_close = prior[-1]
                    elif len(d_closes) >= 2:
                        prev_close = float(d_closes.iloc[-2])
            if prev_close is None:
                prev_close = float(closes.iloc[0])

        if prev_close is None or not math.isfinite(prev_close) or prev_close <= 0:
            return None
        change = current_price - prev_close
        change_pct = ((current_price / prev_close) - 1.0) * 100.0
        return {
            "symbol": sym,
            "current_price": current_price,
            "prev_close": prev_close,
            "change": round(change, 4),
            "change_pct": round(change_pct, 4),
            "source": "yfinance_prepost_15m",
        }

    async def get_us_prepost_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest 15m pre/post-market quotes for US symbols and compute gap vs previous regular close."""
        us_syms = list(
            dict.fromkeys(
                s.strip().upper()
                for s in symbols
                if s and s.strip() and _is_us_symbol(s.strip())
            )
        )
        if not us_syms:
            return {}

        frame_map: dict[str, pd.DataFrame] = {}
        for idx in range(0, len(us_syms), 30):
            chunk = us_syms[idx : idx + 30]
            async with self._sem:
                chunk_res = await asyncio.to_thread(self._download_prepost_15m_sync, chunk)
            frame_map.update(chunk_res)

        missing = [s for s in us_syms if s not in frame_map or frame_map[s].empty]
        if missing:
            async def _fb(sym: str) -> tuple[str, pd.DataFrame]:
                async with self._sem:
                    df = await asyncio.to_thread(self._fetch_prepost_15m_single_sync, sym)
                return sym, df

            for sym, df in await asyncio.gather(*(_fb(s) for s in missing)):
                if not df.empty:
                    frame_map[sym] = df

        out: dict[str, dict[str, Any]] = {}
        for sym in us_syms:
            quote = self._extract_intraday_15m_quote(sym, frame_map.get(sym))
            if quote is not None:
                out[sym] = quote
        return out
