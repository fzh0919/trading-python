"""Quantitative multi-market pre-screener for /pick (0 LLM tokens before Agent review)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from src.engine.screener.patterns import (
    _normalize_regime_tag,
    ScreenerCandidate,
    _linear_scale,
    _extract_ohlcv_series,
    detect_breakthrough_platform,
    detect_low_atr_contraction,
    detect_volume_ma_breakout,
    detect_minervini_stage2,
    detect_vcp_contraction,
    compute_relative_strength,
    compute_vpvr_profile,
    compute_market_breadth,
    compute_portfolio_risk_metrics,
)
from src.engine.screener.scoring import (
    _safe_target_upside_pct,
    _resolve_configured_etfs,
    _format_summary_line,
    score_candidate,
    _apply_percentile_calibration,
)
from src.engine.screener.runner import (
    run_quant_screener,
)
from src.engine.screener.audit import (
    _lookup_history_df,
    _audit_rolling_overlap_candidates,
    _audit_zombie_candidates,
    _backtest_breakout_thresholds,
    _FUNDAMENTAL_COVERAGE_FIELDS,
    compute_fundamentals_coverage,
    run_offline_universe_and_threshold_audit,
)
from src.engine.screener.audition import (
    _approx_usd_multiplier,
    _match_t212_eligibility,
    _is_uk_or_eu_symbol,
    _score_audition_equity,
    run_universe_audition,
)

__all__ = [
    "ScreenerCandidate",
    "detect_breakthrough_platform",
    "detect_low_atr_contraction",
    "detect_volume_ma_breakout",
    "detect_minervini_stage2",
    "detect_vcp_contraction",
    "compute_relative_strength",
    "compute_vpvr_profile",
    "compute_market_breadth",
    "compute_portfolio_risk_metrics",
    "score_candidate",
    "run_quant_screener",
    "compute_fundamentals_coverage",
    "run_offline_universe_and_threshold_audit",
    "run_universe_audition",
]
