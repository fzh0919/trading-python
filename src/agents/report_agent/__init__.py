"""Morning Report (/report) TradingAgents-Lite 3-Step Pipeline & /us Sentinel."""

from __future__ import annotations

import logging

from src.agents.tools import run_forced_market_grounding
from src.engine.calendar_utils import (
    get_market_calendar_alerts,
    get_us_market_schedule_uk,
    is_uk_bank_holiday,
)

logger = logging.getLogger(__name__)

from src.agents.report_agent.fact_sheet import (
    LONDON_TZ,
    CONTEXT_PACK_VERSION,
    _PENDING_PROXIMITY_PCT,
    _VIX_DANGER_LEVEL,
    _CORE_DCA_HORIZON_DAYS,
    _is_us_symbol,
    _extract_holding_context,
    _format_snapshot_age,
    _compute_portfolio_diff_text,
    _account_change_pct,
    _lookup_benchmark_hist,
    _format_benchmark_line,
    _compact_lesson_record,
    _fetch_symbol_lessons,
    build_analysis_context_pack,
    _assemble_fact_sheet_bundle,
    build_portfolio_fact_sheet,
)
from src.agents.report_agent.debate import (
    STEP2_ANALYST_FALLBACK,
    STEP2_AUDITOR_FALLBACK,
    _gather_step2_notes,
    _DEBATE_NEUTRAL_SYSTEM,
    Step2DebateResult,
    _gather_named_notes,
    _build_rebuttal_prompt,
    _build_rebuttal_system,
    _build_debate_digest_prompt,
    _build_factcheck_prompt,
    _build_debate_pm_block,
    _notify_debate_degraded,
    _run_step2_debate,
)
from src.agents.report_agent.core_deployment import (
    _evaluate_deployment_gate_safely,
    _build_report_recommendations,
    _fetch_core_etf_candidates,
    _format_core_policy_ticket_block,
    _extract_gbp_amount,
    _validate_core_deployment,
    _enrich_gate_candidate_bundles,
)
from src.agents.report_agent.sentinels import (
    _daily_move_pct_from_hist,
    _evaluate_us_anomaly_gate,
    generate_us_sentinel,
    generate_intraday_alert,
)
from src.agents.report_agent.morning_report import (
    generate_morning_report,
)

__all__ = [
    "LONDON_TZ",
    "CONTEXT_PACK_VERSION",
    "build_analysis_context_pack",
    "build_portfolio_fact_sheet",
    "STEP2_ANALYST_FALLBACK",
    "STEP2_AUDITOR_FALLBACK",
    "Step2DebateResult",
    "generate_us_sentinel",
    "generate_intraday_alert",
    "generate_morning_report",
]
