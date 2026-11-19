"""HTTP authentication, rate-limit backoff/retry, and order/cash-event fetching mixin for T212Client."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from src.data_providers.t212_client.models import (
    HistoricalDividendItem,
    HistoricalOrderItem,
    HistoricalTransactionItem,
    PendingOrderItem,
)
from src.data_providers.t212_client.wallets_and_tickers import (
    _format_ms_to_uk,
    _parse_iso_ms,
    _parse_ratelimit_remaining,
    _parse_ratelimit_reset_ts,
    clean_t212_ticker,
    to_yahoo_symbol,
)

logger = logging.getLogger("src.data_providers.t212_client")


class T212HttpMixin:
    async def _notify_ops(
        self,
        severity: str,
        alert_key: str,
        message: str,
        *,
        title: str = "",
        business_blocking: bool = False,
        bot_message: str = "",
    ) -> None:
        if self._ops is None:
            return
        if business_blocking and hasattr(self._ops, "notify_business_critical"):
            await self._ops.notify_business_critical(
                source="t212",
                alert_key=alert_key,
                message=message,
                title=title,
                bot_message=bot_message,
            )
            return
        await self._ops.notify(severity=severity, source="t212", alert_key=alert_key, message=message)

    @staticmethod
    def _auth_headers(api_key: str, api_secret: str) -> dict[str, str]:
        token = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("ascii")
        return {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        }

    @staticmethod
    def _compute_rate_limit_sleep(
        headers: httpx.Headers,
        default_sleep_s: float,
        max_wait_s: float = 15.0,
    ) -> float:
        reset_ts = _parse_ratelimit_reset_ts(headers)
        if reset_ts is not None:
            delta = reset_ts - time.time()
            if delta > 0:
                return min(max(delta + 0.05, 0.05), max_wait_s)
        retry_after = headers.get("Retry-After")
        if retry_after:
            try:
                val = float(str(retry_after).strip())
                if val > 0:
                    return min(max(val, 0.05), max_wait_s)
            except ValueError:
                pass
        return min(max(default_sleep_s, 0.05), max_wait_s)

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        default_retry_sleep_s: float = 3.0,
        max_retries: int = 2,
        rate_bucket: str = "default",
        max_rate_wait_s: float = 15.0,
        transient_backoff_s: float | None = None,
    ) -> httpx.Response:
        backoff_base = self._transient_backoff_s if transient_backoff_s is None else transient_backoff_s
        endpoint_key = (rate_bucket, urlparse(url).path)
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            pending_reset = self._rate_limit_resets.get(endpoint_key)
            if pending_reset is not None:
                wait_s = pending_reset - time.time()
                if wait_s > 0:
                    proactive_sleep = min(wait_s + 0.05, max_rate_wait_s)
                    logger.debug("T212 rate-limit pacing on %s (%s), sleeping %.2fs", url, rate_bucket, proactive_sleep)
                    await asyncio.sleep(proactive_sleep)
                self._rate_limit_resets.pop(endpoint_key, None)

            try:
                resp = await client.get(url, headers=headers, timeout=20.0)
                remaining = _parse_ratelimit_remaining(resp.headers)
                reset_ts = _parse_ratelimit_reset_ts(resp.headers)
                if remaining is not None and remaining <= 0 and reset_ts is not None:
                    self._rate_limit_resets[endpoint_key] = reset_ts
                elif resp.status_code == 200 and (remaining is None or remaining > 0):
                    self._rate_limit_resets.pop(endpoint_key, None)

                if resp.status_code == 429:
                    if attempt == max_retries:
                        return resp
                    sleep_s = self._compute_rate_limit_sleep(resp.headers, default_retry_sleep_s, max_wait_s=max_rate_wait_s)
                    self._rate_limit_resets.pop(endpoint_key, None)
                    logger.warning("T212 429 rate-limited on %s, waiting %.2fs", url, sleep_s)
                    await asyncio.sleep(sleep_s)
                    continue

                if resp.status_code == 408 or (500 <= resp.status_code < 600):
                    if attempt == max_retries:
                        return resp
                    backoff = min(backoff_base * (attempt + 1), 3.0)
                    logger.warning("T212 HTTP %d on %s, retrying in %.2fs", resp.status_code, url, backoff)
                    await asyncio.sleep(backoff)
                    continue

                return resp
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == max_retries:
                    raise
                await asyncio.sleep(min(backoff_base * (attempt + 1), 3.0))
        assert last_exc is not None
        raise last_exc

    async def _fetch_order_data(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        account_label: str,
        account_currency: str,
    ) -> tuple[list[PendingOrderItem], bool, str, list[HistoricalOrderItem], bool]:
        pending_orders: list[PendingOrderItem] = []
        pending_supported = False
        pending_error = ""
        recent_orders: list[HistoricalOrderItem] = []
        history_supported = False

        # 1. Active pending orders (/equity/orders)
        try:
            active_res = await self._request_with_retry(
                client,
                f"{self.base_url}/equity/orders",
                headers,
                default_retry_sleep_s=5.2,
                rate_bucket=account_label,
            )
            if active_res.status_code == 200:
                pending_supported = True
                raw_pending = active_res.json()
                if isinstance(raw_pending, list):
                    for o in raw_pending:
                        inst = o.get("instrument") or {}
                        raw_ticker = str(o.get("ticker") or inst.get("ticker") or "UNKNOWN")
                        # Note: T212 Order.currency is often the account settlement currency (e.g. GBP)
                        # whereas limitPrice/stopPrice are quoted in the instrument's trading currency.
                        clean, inst_curr, yahoo_sym, _ = self._resolve_ticker_metadata(
                            raw_ticker,
                            inst,
                            account_currency,
                            extra_currency=str(o.get("currency") or ""),
                        )
                        signed_qty = float(o.get("quantity") or 0)
                        raw_qty = abs(signed_qty)
                        filled_qty = abs(float(o.get("filledQuantity") or 0))
                        remaining_qty = max(raw_qty - filled_qty, 0.0) if (raw_qty > 0 and filled_qty > 0) else raw_qty
                        raw_val = abs(float(o.get("value") or 0))
                        filled_val = abs(float(o.get("filledValue") or 0))
                        remaining_val = max(raw_val - filled_val, 0.0) if (raw_val > 0 and filled_val > 0) else raw_val
                        order_status = str(o.get("status") or "")
                        side = str(o.get("side") or ("SELL" if signed_qty < 0 else "BUY"))
                        pending_orders.append(
                            PendingOrderItem(
                                id=str(o.get("id") or ""),
                                account=account_label,
                                raw_ticker=raw_ticker,
                                ticker=clean,
                                yahoo_symbol=yahoo_sym,
                                name=str(inst.get("name") or clean),
                                side=side,
                                order_type=str(o.get("type") or "LIMIT"),
                                quantity=remaining_qty,
                                filled_quantity=filled_qty,
                                status=order_status,
                                value=remaining_val,
                                limit_price=float(o["limitPrice"]) if o.get("limitPrice") is not None else None,
                                stop_price=float(o["stopPrice"]) if o.get("stopPrice") is not None else None,
                                currency=inst_curr,
                                time_in_force=str(o.get("timeInForce") or ""),
                                created_at=str(o.get("createdAt") or ""),
                            )
                        )
            elif active_res.status_code == 403:
                logger.info("%s /equity/orders returned 403 (Orders - Read disabled).", account_label)
                await self._notify_ops(
                    "WARN",
                    f"t212_orders_403:{account_label}",
                    f"Trading 212 [{account_label}] 活动挂单接口返回 403（可能缺少 Orders - Read 权限或被风控限制）。",
                )
            else:
                pending_error = f"HTTP {active_res.status_code}"
                logger.warning("%s /equity/orders HTTP %d", account_label, active_res.status_code)
                await self._notify_ops(
                    "WARN",
                    f"t212_orders_http:{account_label}",
                    f"Trading 212 [{account_label}] 活动挂单接口返回 HTTP {active_res.status_code}。",
                )
        except Exception as exc:
            pending_error = type(exc).__name__
            logger.warning("%s failed fetching pending orders: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_orders_exc:{account_label}",
                f"Trading 212 [{account_label}] 活动挂单拉取异常：{type(exc).__name__}: {exc}",
            )

        # 2. Historical filled orders (/equity/history/orders, 30d window)
        try:
            cutoff_ms = int(time.time() * 1000) - self.order_history_days * 86400 * 1000
            origin = re.sub(r"/api/v0/?$", "", self.base_url)
            next_url: str | None = f"{self.base_url}/equity/history/orders?limit=50"
            seen_ids: set[str] = set()

            for page in range(self.order_history_max_pages):
                if not next_url:
                    break
                hist_res = await self._request_with_retry(
                    client,
                    next_url,
                    headers,
                    default_retry_sleep_s=3.2,
                    rate_bucket=account_label,
                )
                if hist_res.status_code == 403:
                    logger.info("%s /equity/history/orders returned 403 (History - Orders disabled).", account_label)
                    await self._notify_ops(
                        "WARN",
                        f"t212_history_403:{account_label}",
                        f"Trading 212 [{account_label}] 历史成交接口返回 403（可能缺少 History - Orders 权限）。",
                    )
                    break
                if hist_res.status_code != 200:
                    logger.warning("%s /equity/history/orders page %d HTTP %d", account_label, page + 1, hist_res.status_code)
                    await self._notify_ops(
                        "WARN",
                        f"t212_history_http:{account_label}",
                        f"Trading 212 [{account_label}] 历史成交接口返回 HTTP {hist_res.status_code}。",
                    )
                    break

                history_supported = True
                body: dict[str, Any] = hist_res.json() or {}
                items = body.get("items") or []
                found_older = False

                for entry in items:
                    order = entry.get("order") or {}
                    fill = entry.get("fill") or None
                    if not fill or not fill.get("filledAt"):
                        continue
                    if fill.get("type") and fill.get("type") != "TRADE":
                        continue

                    raw_qty = fill.get("quantity") if fill.get("quantity") is not None else order.get("filledQuantity", 0)
                    qty = abs(float(raw_qty or 0))
                    if qty <= 0:
                        continue

                    filled_at_ms = _parse_iso_ms(str(fill["filledAt"]))
                    if filled_at_ms <= 0:
                        continue
                    if filled_at_ms < cutoff_ms:
                        found_older = True
                        continue

                    dedup_id = str(fill.get("id") or f"{order.get('id', '')}_{fill['filledAt']}")
                    if dedup_id in seen_ids:
                        continue
                    seen_ids.add(dedup_id)

                    inst = order.get("instrument") or {}
                    wallet = fill.get("walletImpact") or {}
                    raw_ticker = str(order.get("ticker") or inst.get("ticker") or "UNKNOWN")
                    wallet_currency = str(wallet.get("currency") or account_currency)
                    clean, inst_curr, yahoo_sym, _ = self._resolve_ticker_metadata(
                        raw_ticker,
                        inst,
                        wallet_currency,
                        extra_currency=str(order.get("currency") or ""),
                    )
                    signed_qty = fill.get("quantity") if fill.get("quantity") is not None else order.get("quantity", 0)
                    side = str(order.get("side") or ("SELL" if float(signed_qty or 0) < 0 else "BUY"))

                    tax_total = 0.0
                    for tax in (wallet.get("taxes") or []):
                        t_curr = tax.get("currency")
                        if not t_curr or t_curr == wallet_currency:
                            tax_total += abs(float(tax.get("quantity") or 0))

                    recent_orders.append(
                        HistoricalOrderItem(
                            id=dedup_id,
                            order_id=str(order.get("id") or "").strip(),
                            account=account_label,
                            raw_ticker=raw_ticker,
                            ticker=clean,
                            yahoo_symbol=yahoo_sym,
                            name=str(inst.get("name") or clean),
                            inst_currency=inst_curr,
                            side=side,
                            order_type=str(order.get("type") or "MARKET"),
                            filled_at=str(fill["filledAt"]),
                            filled_at_ms=filled_at_ms,
                            uk_time_str=_format_ms_to_uk(filled_at_ms),
                            quantity=qty,
                            price=float(fill.get("price") or 0),
                            net_value=abs(float(wallet.get("netValue") or 0)),
                            wallet_currency=wallet_currency,
                            realised_pnl=float(wallet.get("realisedProfitLoss") or 0),
                            tax_total=tax_total,
                        )
                    )

                next_path = body.get("nextPagePath")
                if found_older or not next_path:
                    break
                next_path_str = str(next_path).strip()
                next_url = (
                    next_path_str
                    if next_path_str.startswith("http")
                    else f"{origin}{'' if next_path_str.startswith('/') else '/'}{next_path_str}"
                )
        except Exception as exc:
            logger.warning("%s failed fetching order history: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_history_exc:{account_label}",
                f"Trading 212 [{account_label}] 历史成交拉取异常：{type(exc).__name__}: {exc}",
            )

        return pending_orders, pending_supported, pending_error, recent_orders, history_supported

    async def _fetch_cash_events(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        account_label: str,
        account_currency: str,
    ) -> tuple[list[HistoricalTransactionItem], bool, list[HistoricalDividendItem], bool]:
        """Fetch 30-day external deposits/withdrawals/interest (/equity/history/transactions) and dividends (/equity/history/dividends)."""
        recent_transactions: list[HistoricalTransactionItem] = []
        tx_supported = False
        recent_dividends: list[HistoricalDividendItem] = []
        div_supported = False
        cutoff_ms = int(time.time() * 1000) - self.order_history_days * 86400 * 1000
        origin = re.sub(r"/api/v0/?$", "", self.base_url)

        # 1. Transactions (/equity/history/transactions)
        try:
            next_url: str | None = f"{self.base_url}/equity/history/transactions?limit=50"
            seen_tx_ids: set[str] = set()
            for page in range(self.order_history_max_pages):
                if not next_url:
                    break
                tx_res = await self._request_with_retry(
                    client,
                    next_url,
                    headers,
                    default_retry_sleep_s=3.2,
                    rate_bucket=account_label,
                )
                if tx_res.status_code == 403:
                    logger.info("%s /equity/history/transactions returned 403 (History - Transactions disabled).", account_label)
                    await self._notify_ops(
                        "WARN",
                        f"t212_tx_403:{account_label}",
                        f"Trading 212 [{account_label}] 出入金流水接口返回 403（可能缺少 History - Transactions 权限）。",
                    )
                    break
                if tx_res.status_code == 404:
                    logger.debug("%s /equity/history/transactions returned 404; skipping.", account_label)
                    break
                if tx_res.status_code != 200:
                    logger.warning("%s /equity/history/transactions page %d HTTP %d", account_label, page + 1, tx_res.status_code)
                    await self._notify_ops(
                        "WARN",
                        f"t212_tx_http:{account_label}",
                        f"Trading 212 [{account_label}] 出入金流水接口返回 HTTP {tx_res.status_code}。",
                    )
                    break

                tx_supported = True
                body: dict[str, Any] = tx_res.json() or {}
                items = body.get("items") or []
                found_older = False
                for t in items:
                    dt_str = str(t.get("dateTime") or "")
                    dt_ms = _parse_iso_ms(dt_str)
                    if dt_ms <= 0:
                        continue
                    if dt_ms < cutoff_ms:
                        found_older = True
                        continue
                    tx_type = str(t.get("type") or "DEPOSIT").upper()
                    raw_amt = float(t.get("amount") or 0)
                    if tx_type in {"WITHDRAW", "FEE"}:
                        signed_amt = -abs(raw_amt)
                    elif tx_type in {"DEPOSIT", "INTEREST_ON_FREE_CASH", "LENDING_INTEREST"}:
                        signed_amt = abs(raw_amt)
                    else:
                        signed_amt = raw_amt
                    dedup_id = str(t.get("reference") or f"{tx_type}_{dt_str}_{signed_amt}")
                    if dedup_id in seen_tx_ids:
                        continue
                    seen_tx_ids.add(dedup_id)
                    recent_transactions.append(
                        HistoricalTransactionItem(
                            id=dedup_id,
                            account=account_label,
                            tx_type=tx_type,
                            amount=signed_amt,
                            currency=str(t.get("currency") or account_currency),
                            date_time=dt_str,
                            date_time_ms=dt_ms,
                            uk_time_str=_format_ms_to_uk(dt_ms),
                        )
                    )
                next_path = body.get("nextPagePath")
                if found_older or not next_path:
                    break
                next_path_str = str(next_path).strip()
                next_url = (
                    next_path_str
                    if next_path_str.startswith("http")
                    else f"{origin}{'' if next_path_str.startswith('/') else '/'}{next_path_str}"
                )
        except Exception as exc:
            logger.warning("%s failed fetching transactions: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_tx_exc:{account_label}",
                f"Trading 212 [{account_label}] 出入金流水拉取异常：{type(exc).__name__}: {exc}",
            )

        # 2. Dividends (/equity/history/dividends)
        try:
            next_url = f"{self.base_url}/equity/history/dividends?limit=50"
            seen_div_ids: set[str] = set()
            for page in range(self.order_history_max_pages):
                if not next_url:
                    break
                div_res = await self._request_with_retry(
                    client,
                    next_url,
                    headers,
                    default_retry_sleep_s=3.2,
                    rate_bucket=account_label,
                )
                if div_res.status_code == 403:
                    logger.info("%s /equity/history/dividends returned 403 (History - Dividends disabled).", account_label)
                    await self._notify_ops(
                        "WARN",
                        f"t212_div_403:{account_label}",
                        f"Trading 212 [{account_label}] 派息记录接口返回 403（可能缺少 History - Dividends 权限）。",
                    )
                    break
                if div_res.status_code == 404:
                    logger.debug("%s /equity/history/dividends returned 404; skipping.", account_label)
                    break
                if div_res.status_code != 200:
                    logger.warning("%s /equity/history/dividends page %d HTTP %d", account_label, page + 1, div_res.status_code)
                    await self._notify_ops(
                        "WARN",
                        f"t212_div_http:{account_label}",
                        f"Trading 212 [{account_label}] 派息记录接口返回 HTTP {div_res.status_code}。",
                    )
                    break

                div_supported = True
                body = div_res.json() or {}
                items = body.get("items") or []
                found_older = False
                for d in items:
                    paid_on = str(d.get("paidOn") or "")
                    paid_ms = _parse_iso_ms(paid_on)
                    if paid_ms <= 0:
                        continue
                    if paid_ms < cutoff_ms:
                        found_older = True
                        continue
                    inst = d.get("instrument") or {}
                    raw_ticker = str(d.get("ticker") or inst.get("ticker") or "UNKNOWN")
                    div_curr = str(d.get("currency") or account_currency)
                    meta = self._instrument_map.get(raw_ticker) or {}
                    inst_curr = str(d.get("tickerCurrency") or inst.get("currency") or meta.get("currency_code") or "")
                    clean = clean_t212_ticker(raw_ticker)
                    yahoo_sym = str(meta.get("yahoo_symbol") or "").strip() or to_yahoo_symbol(
                        raw_ticker, inst_curr or div_curr, clean
                    )
                    dedup_id = str(d.get("reference") or f"{raw_ticker}_{paid_on}")
                    if dedup_id in seen_div_ids:
                        continue
                    seen_div_ids.add(dedup_id)
                    recent_dividends.append(
                        HistoricalDividendItem(
                            id=dedup_id,
                            account=account_label,
                            raw_ticker=raw_ticker,
                            ticker=clean,
                            yahoo_symbol=yahoo_sym,
                            name=str(inst.get("name") or clean),
                            inst_currency=inst_curr,
                            dividend_type=str(d.get("type") or "DIVIDEND"),
                            amount=float(d.get("amount") or 0),
                            currency=div_curr,
                            gross_per_share=float(d.get("grossAmountPerShare") or 0),
                            quantity=float(d.get("quantity") or 0),
                            paid_on=paid_on,
                            paid_on_ms=paid_ms,
                            uk_time_str=_format_ms_to_uk(paid_ms),
                        )
                    )
                next_path = body.get("nextPagePath")
                if found_older or not next_path:
                    break
                next_path_str = str(next_path).strip()
                next_url = (
                    next_path_str
                    if next_path_str.startswith("http")
                    else f"{origin}{'' if next_path_str.startswith('/') else '/'}{next_path_str}"
                )
        except Exception as exc:
            logger.warning("%s failed fetching dividends: %s", account_label, exc)
            await self._notify_ops(
                "WARN",
                f"t212_div_exc:{account_label}",
                f"Trading 212 [{account_label}] 派息记录拉取异常：{type(exc).__name__}: {exc}",
            )

        return recent_transactions, tx_supported, recent_dividends, div_supported
