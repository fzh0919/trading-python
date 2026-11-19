"""Timezone-aware weekday scheduler (Europe/London + US market DST/holiday adaptation) and post-close settlement."""

from __future__ import annotations

import logging

from src.agents.report_agent import generate_morning_report
from src.engine.screener import run_universe_audition

logger = logging.getLogger(__name__)

from src.bot.scheduler.settlement_math import (
    LONDON_TZ,
    MORNING_CATCHUP_END_MINUTES,
    DAILY_SETTLEMENT_START_MINUTES,
    DAILY_SETTLEMENT_END_MINUTES,
    NAV_SNAPSHOT_START_MINUTES,
    NAV_SNAPSHOT_END_MINUTES,
    WEEKLY_AUDIT_START_MINUTES,
    WEEKLY_AUDIT_END_MINUTES,
    INTRADAY_SENTINEL_START_MINUTES,
    INTRADAY_SENTINEL_END_MINUTES,
    WEEKLY_WALLET_RECONCILE_START_MINUTES,
    _parse_recommendation_date,
    _resolve_benchmark_candidates,
    _extract_closed_daily_ohlc,
    _extract_closed_daily_closes,
    _align_lse_unit_100x,
    _align_entry_price_gbp_guard,
    _trade_direction,
    _direction_adjusted_return_pct,
    _direction_adjusted_alpha_pct,
    _entry_fill_status,
    _compute_mfe_mae_pct,
    _match_recommendation_fill,
    _compute_benchmark_return_pct,
    _clean_reflection_lesson,
    _normalize_quote,
    _quote_is_verbatim,
)
from src.bot.scheduler.fill_receipts import (
    _to_float,
    _intraday_move_pct,
    _order_distance_pct,
    _normalize_account_label,
    _has_protective_stop,
    _close_series_from_hist,
    _drawdown_from_high_pct,
    _atr_pct_from_hist,
    _invest_adaptive_stop_pct,
    _format_pending_orders,
    FILL_QTY_TOLERANCE,
    FILL_MATCH_GRACE_MS,
    FILL_RECEIPT_MAX_CHARS,
    _pending_order_snapshot_row,
    _snapshot_key,
    diff_pending_order_snapshots,
    match_fill_history,
    _format_uk_from_ms,
    format_fill_receipt,
)
from src.bot.scheduler.core import (
    TradingScheduler,
)

__all__ = [
    "LONDON_TZ",
    "MORNING_CATCHUP_END_MINUTES",
    "DAILY_SETTLEMENT_START_MINUTES",
    "DAILY_SETTLEMENT_END_MINUTES",
    "NAV_SNAPSHOT_START_MINUTES",
    "NAV_SNAPSHOT_END_MINUTES",
    "WEEKLY_AUDIT_START_MINUTES",
    "WEEKLY_AUDIT_END_MINUTES",
    "INTRADAY_SENTINEL_START_MINUTES",
    "INTRADAY_SENTINEL_END_MINUTES",
    "WEEKLY_WALLET_RECONCILE_START_MINUTES",
    "FILL_QTY_TOLERANCE",
    "FILL_MATCH_GRACE_MS",
    "FILL_RECEIPT_MAX_CHARS",
    "diff_pending_order_snapshots",
    "match_fill_history",
    "format_fill_receipt",
    "TradingScheduler",
]
