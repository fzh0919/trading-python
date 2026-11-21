"""0-Token post-close execution ledger, catalyst budget, and silent Flash Lite lookup mixin."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.bot.scheduler.settlement_math import LONDON_TZ
from src.data_providers.analytics import (
    build_eod_attribution_ledger,
    format_postmarket_alert_text,
    format_postmarket_review_text,
    is_substantive_catalyst,
    resolve_market_benchmark,
)

logger = logging.getLogger("src.bot.scheduler")


def _recent_trading_day_cutoff(uk_date: str, trading_days: int = 2) -> date | None:
    """Return the date N trading weekdays before ``uk_date`` (holiday-free approximation)."""
    try:
        cursor = date.fromisoformat(str(uk_date or "")[:10])
    except ValueError:
        return None
    remaining = max(0, int(trading_days))
    while remaining > 0:
        cursor -= timedelta(days=1)
        if cursor.weekday() < 5:
            remaining -= 1
    return cursor


class PostmarketReviewMixin:
    def _postmarket_search_budget(self) -> int:
        """Single-night Flash Lite search budget; disabled switch or 0 means no search at all."""
        if not bool(getattr(self.cfg.strategy, "postmarket_event_search_enabled", True)):
            return 0
        try:
            configured = int(getattr(self.cfg.strategy, "postmarket_max_search_symbols", 3) or 0)
        except (TypeError, ValueError):
            configured = 3
        return max(0, configured)

    @staticmethod
    def _clean_catalyst_text(raw: Any) -> str:
        text = (
            str(raw or "")
            .replace("#", "")
            .replace("**", "")
            .replace("\r", " ")
            .replace("\n", " ")
        )
        text = " ".join(text.split())
        if not text or text.startswith("⚠️"):
            return ""
        return text[:80]

    async def _fetch_catalyst_with_flash_lite(
        self,
        symbol: str,
        *,
        window_label: str,
        reason_hint: str,
        db: Any,
        cache_scope: str = "",
        uk_date: str = "",
    ) -> str:
        """Best-effort objective catalyst lookup; hard 30s cap and silent degradation.

        Reuses the existing ``gemini-3.5-flash-lite`` search route and the 3h
        ``grounding_news_cache``.  Any failure returns an empty string and must never
        block settlement, pruning, or the rest of the postmarket ledger.
        """
        sym = str(symbol or "").strip().upper()
        if not sym:
            return ""
        day_key = (
            str(uk_date or "").strip()[:10]
            or datetime.now(timezone.utc).astimezone(LONDON_TZ).date().isoformat()
        )
        raw_scope = str(cache_scope or window_label or "default").strip().lower()
        scope_slug = (
            "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw_scope)[:32].strip("_")
            or "default"
        )
        cache_key = f"postmarket_catalyst:{day_key}:{scope_slug}:{sym}"
        try:
            if db is not None and hasattr(db, "get_grounding_news_cache"):
                cached = await db.get_grounding_news_cache([cache_key], max_age_seconds=10800)
                hit = str((cached or {}).get(cache_key) or "").strip()
                if hit:
                    return self._clean_catalyst_text(hit)
            prompt = (
                f"Ticker: {sym}\n"
                f"Window: {window_label}\n"
                f"Context: {str(reason_hint or '').strip()[:240]}\n"
                "Retrieve only objective corporate announcements in this window."
            )
            system_instruction = (
                "You are a financial event researcher. Use Google Search to verify primary English "
                "financial and regulatory sources (RNS, SEC, exchange disclosures, company IR). "
                "Extract ONLY official earnings/guidance, M&A, regulatory/FDA decisions, major bank "
                "rating changes, or dividend/buyback changes. Never speculate, never pad with price "
                "recaps. Output <=80 Chinese characters of plain text, no # or **. If there is no "
                "explicit announcement, reply exactly: 无明确基本面公告，属市场/板块波动."
            )
            raw = await self.llm.generate(
                model=self.cfg.models.search_model,
                prompt=prompt,
                system_instruction=system_instruction,
                thinking_level="LOW",
                use_google_search=True,
                temperature=0.1,
                usage_label="postmarket_catalyst",
                round_timeout_seconds=30.0,
                max_output_tokens=512,
            )
            text = self._clean_catalyst_text(raw)
            if text and db is not None and hasattr(db, "upsert_grounding_news_cache"):
                try:
                    await db.upsert_grounding_news_cache(
                        [
                            {
                                "cache_key": cache_key,
                                "scope": "postmarket",
                                "symbol": sym,
                                "summary_text": text,
                                "fetched_at_ms": int(time.time() * 1000),
                            }
                        ]
                    )
                except Exception as cache_exc:
                    logger.debug("Postmarket catalyst cache write failed for %s: %s", sym, cache_exc)
            return text
        except Exception as exc:
            logger.warning("Postmarket catalyst lookup skipped for %s: %s", sym, exc)
            return ""

    async def _run_postmarket_review(
        self,
        *,
        now_utc: datetime,
        uk_date: str,
        search_budget: int,
    ) -> str:
        """Build, persist, and optionally alert on the 0-Token EOD execution ledger."""
        if not bool(getattr(self.cfg.strategy, "postmarket_review_enabled", True)):
            return ""
        # Weekdays only (same contract as nav_snapshot): Saturday/Sunday daily_settlement
        # must never replay Friday's prices into duplicate weekend rows or repeat alerts.
        if now_utc.astimezone(LONDON_TZ).weekday() >= 5:
            return ""
        try:
            portfolio = await self.t212.get_portfolio(use_cache=True)
        except Exception as exc:
            logger.warning("Postmarket review portfolio fetch failed: %s", exc)
            return ""
        if portfolio is None or getattr(portfolio, "is_fallback", False):
            return ""
        if float(getattr(portfolio, "total_combined_value", 0.0) or 0.0) <= 0.0:
            return ""

        morning = None
        try:
            if hasattr(self.db, "get_snapshot_for_cycle_date"):
                morning = await self.db.get_snapshot_for_cycle_date(uk_date)
            if morning is None:
                morning = await self.db.get_previous_snapshot(min_age_hours=6.0)
        except Exception as exc:
            logger.debug("Postmarket morning baseline unavailable: %s", exc)
            morning = None
        if morning is None or getattr(morning, "is_fallback", False):
            return ""
        # If the earliest snapshot for this cycle is already the EOD fetch, there is no
        # morning baseline; fall back to the latest snapshot at least 6h old.
        try:
            if int(getattr(morning, "fetched_at_ms", 0) or 0) >= int(
                getattr(portfolio, "fetched_at_ms", 0) or 0
            ):
                morning = await self.db.get_previous_snapshot(min_age_hours=6.0)
        except Exception as exc:
            logger.debug("Postmarket baseline ordering check failed: %s", exc)
        if morning is None or getattr(morning, "is_fallback", False):
            return ""
        # get_previous_snapshot falls back to the newest row when none is >=6h old; refuse
        # to self-diff the EOD snapshot against itself and fabricate a £0.00 ledger.
        try:
            if int(getattr(morning, "fetched_at_ms", 0) or 0) >= int(
                getattr(portfolio, "fetched_at_ms", 0) or 0
            ):
                return ""
        except Exception:
            return ""

        recent_recs: list[dict[str, Any]] = []
        try:
            rows = await self.db.get_recent_recommendations(limit=30)
            cutoff = _recent_trading_day_cutoff(uk_date, trading_days=2)
            for row in rows or []:
                raw_date = str(row.get("created_at_uk") or "")[:10]
                try:
                    rec_day = date.fromisoformat(raw_date)
                except ValueError:
                    continue
                if cutoff is None or rec_day >= cutoff:
                    recent_recs.append(row)
        except Exception as exc:
            logger.debug("Postmarket recommendations unavailable: %s", exc)

        regime = ""
        try:
            for item in await self.db.get_regime_history(limit=5):
                if str(item.get("uk_date") or "") <= str(uk_date)[:10]:
                    regime = str(item.get("regime") or "")
        except Exception as exc:
            logger.debug("Postmarket regime label unavailable: %s", exc)

        symbols: list[str] = []
        for pos in portfolio.all_positions():
            sym = str(getattr(pos, "yahoo_symbol", "") or "").strip().upper()
            if sym:
                symbols.append(sym)
                symbols.extend(resolve_market_benchmark(sym))
        for rec in recent_recs:
            sym = str(rec.get("symbol") or "").strip().upper()
            if sym:
                symbols.append(sym)
        for attr in ("benchmark_symbol", "regime_benchmark_symbol"):
            raw = str(getattr(self.cfg.strategy, attr, "") or "").strip().upper()
            if raw:
                symbols.append(raw)
        histories: dict[str, Any] = {}
        if symbols:
            try:
                histories = await self.market_data.get_histories(
                    list(dict.fromkeys(symbols)), period="1y"
                )
            except Exception as exc:
                logger.debug("Postmarket histories unavailable: %s", exc)

        ledger = build_eod_attribution_ledger(
            morning,
            portfolio,
            regime=regime,
            today_recs=recent_recs,
            histories=histories,
            strategy=self.cfg.strategy,
            uk_date=uk_date,
        )

        event_cards: list[dict[str, Any]] = []
        remaining = max(0, int(search_budget))
        for candidate in ledger.get("anomaly_candidates") or []:
            if remaining <= 0:
                break
            remaining -= 1
            text = await self._fetch_catalyst_with_flash_lite(
                str(candidate.get("symbol") or ""),
                window_label=f"post-close anomaly ({uk_date})",
                reason_hint=str(candidate.get("event_reason_hint") or ""),
                db=self.db,
                cache_scope="eod_anomaly",
                uk_date=uk_date,
            )
            if is_substantive_catalyst(text):
                event_cards.append(
                    {
                        "symbol": str(candidate.get("symbol") or ""),
                        "text": text,
                        "account": str(candidate.get("account") or ""),
                    }
                )

        summary_text = format_postmarket_review_text(ledger, event_cards=event_cards, max_chars=900)
        alert_text = format_postmarket_alert_text(ledger, event_cards=event_cards, max_chars=250)
        if not bool(getattr(self.cfg.strategy, "postmarket_push_on_alert_only", True)):
            alert_text = summary_text[:250]
        if not summary_text:
            return ""

        try:
            await self.db.upsert_daily_postmarket_review(
                {
                    "uk_date": uk_date,
                    "created_at_uk": now_utc.astimezone(LONDON_TZ).strftime("%Y-%m-%d %H:%M"),
                    "created_at_ms": int(now_utc.timestamp() * 1000),
                    "regime": ledger.get("regime") or "",
                    "local_pnl_gbp": (ledger.get("totals") or {}).get("local_pnl_gbp", 0.0),
                    "fx_pnl_gbp": (ledger.get("totals") or {}).get("fx_pnl_gbp", 0.0),
                    "total_mark_pnl_gbp": (ledger.get("totals") or {}).get(
                        "total_mark_pnl_gbp", 0.0
                    ),
                    "discipline_alert_count": int(ledger.get("discipline_alert_count") or 0),
                    "summary_text": summary_text,
                    "payload": {**ledger, "event_cards": event_cards},
                }
            )
        except Exception as exc:
            logger.warning("Postmarket review persistence failed: %s", exc)
            return ""
        return alert_text
