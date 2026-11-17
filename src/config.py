"""Configuration loader combining .env secrets and config.yaml strategy settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)
ThinkingLevelLiteral = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"]


def _normalize_symbol_list(raw: Any) -> list[str]:
    """Strip, uppercase, drop blanks, and deduplicate ticker symbols preserving order."""
    if not raw:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in raw:
        sym = str(item or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            cleaned.append(sym)
    return cleaned


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    t212_isa_key: str = ""
    t212_isa_secret: str = ""
    t212_invest_key: str = ""
    t212_invest_secret: str = ""
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    finnhub_api_key: str = ""
    fred_api_key: str = ""
    t212_base_url: str = "https://live.trading212.com/api/v0"
    ops_notify_url: str = ""
    ops_notify_token: str = ""


class AppSection(BaseModel):
    timezone: str = "Europe/London"
    db_path: str = "data/advisor.db"
    log_level: str = "INFO"
    cache_ttl_seconds: int = 120
    market_data_ttl_seconds: int = 900
    grounding_cache_ttl_seconds: int = Field(default=10800, ge=0)
    ops_warn_cooldown_seconds: int = 1800


class ModelsSection(BaseModel):
    # Hybrid Gemini + DeepSeek routing with cross-vendor failover
    report_analyst_model: str = "gemini-4-hydrogen"
    report_auditor_model: str = "gemini-4-carbon"
    report_model: str = "gemini-4-hydrogen"
    us_sentinel_model: str = "deepseek-flash"
    us_anomaly_model: str = "gemini-4-hydrogen"
    pick_researcher_model: str = "gemini-4-hydrogen"
    pick_model: str = "gemini-4-hydrogen"
    chat_model: str = "deepseek-flash"
    chat_deep_model: str = "gemini-4-hydrogen"
    fallback_model: str = "gemini-4-hydrogen"
    gemini_fallback_model: str = "gemini-4-hydrogen"
    deepseek_fallback_model: str = "deepseek-flash"
    report_priority_search_model: str = "gemini-4-hydrogen"
    report_search_model: str = "gemini-3.5-flash-lite"
    search_model: str = "gemini-3.5-flash-lite"
    # DeepSeek V4.1 Flash sidecar (ID filter / read-only linter / independent briefs / quoted memory)
    ds_helper_model: str = "deepseek-flash"
    ds_helper_thinking_level: ThinkingLevelLiteral = "HIGH"
    ds_helper_timeout_seconds: float = Field(default=20.0, ge=5.0, le=60.0)
    ds_pick_prefilter_enabled: bool = True
    ds_semantic_audit_enabled: bool = True

    report_analyst_thinking_level: ThinkingLevelLiteral = "HIGH"
    report_auditor_thinking_level: ThinkingLevelLiteral = "HIGH"
    report_thinking_level: ThinkingLevelLiteral = "HIGH"
    us_thinking_level: ThinkingLevelLiteral = "HIGH"
    us_anomaly_thinking_level: ThinkingLevelLiteral = "HIGH"
    pick_researcher_thinking_level: ThinkingLevelLiteral = "HIGH"
    pick_thinking_level: ThinkingLevelLiteral = "HIGH"
    chat_thinking_level: ThinkingLevelLiteral = "HIGH"
    chat_deep_thinking_level: ThinkingLevelLiteral = "HIGH"
    fallback_thinking_level: ThinkingLevelLiteral = "HIGH"
    gemini_fallback_thinking_level: ThinkingLevelLiteral = "MEDIUM"
    deepseek_fallback_thinking_level: ThinkingLevelLiteral = "HIGH"
    report_priority_search_thinking_level: ThinkingLevelLiteral = "LOW"

    enable_forced_search_grounding: bool = True
    report_max_tool_rounds: int = Field(default=2, ge=1)
    google_search_free_quota_monthly: int = Field(default=5000, ge=0)
    google_search_warn_pct: float = Field(default=85.0, gt=0.0, le=100.0)

    # Legacy v5.0 keys retained for smooth backwards compatibility
    premarket_model: str | None = None
    fast_thinking_level: ThinkingLevelLiteral | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_model_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "premarket_model" in out and out["premarket_model"] and "report_analyst_model" not in out:
            out["report_analyst_model"] = out["premarket_model"]
        if "fast_thinking_level" in out and out["fast_thinking_level"]:
            legacy_fast = out["fast_thinking_level"]
            out.setdefault("report_analyst_thinking_level", legacy_fast)
            out.setdefault("us_thinking_level", legacy_fast)
            out.setdefault("chat_thinking_level", legacy_fast)
        return out


class ScheduleSection(BaseModel):
    morning_report_hour: int = 7
    morning_report_minute: int = 30
    premarket_lead_minutes: int = 15
    enable_weekly_radar: bool = True
    weekly_radar_after_close_minutes: int = 30


class StrategySection(BaseModel):
    cash_interest_rate_pct: float = 4.0
    usd_cash_interest_rate_pct: float = 3.4
    fx_fee_pct: float = 0.15
    uk_stamp_duty_pct: float = 0.50
    # Invest GBP+USD dual-wallet ledger (T212 API v0 only reports GBP-converted total cash)
    invest_multi_currency_enabled: bool = True
    wallet_reconcile_warn_days: int = Field(default=7, ge=0)
    wallet_mismatch_warn_gbp: float = Field(default=10.0, ge=0.0)
    order_history_days: int = 30
    order_history_max_pages: int = 3
    max_daily_chat_turns: int = 12
    # Deterministic Add/Trim/Stop/Correlation guardrails (used by portfolio_rules.py & screener.py)
    max_isa_single_weight_pct: float = 30.0
    max_invest_single_weight_pct: float = 25.0
    max_total_portfolio_weight_pct: float = 22.0
    invest_stop_loss_alert_pct: float = -12.0
    invest_take_profit_review_pct: float = 25.0
    trailing_profit_activation_pct: float = 20.0
    trailing_drawdown_from_peak_pct: float = -12.0
    isa_deep_drawdown_review_pct: float = -25.0
    rsi_oversold_threshold: float = 38.0
    rsi_overbought_threshold: float = 75.0
    pullback_from_high_add_min_pct: float = -25.0
    pullback_from_high_add_max_pct: float = -8.0
    require_uptrend_for_add: bool = True
    overlap_correlation_threshold: float = 0.85
    correlation_window_days: int = 60
    # /us intraday anomaly gate & T+N settlement horizon
    us_anomaly_move_pct: float = Field(default=1.8, gt=0.0)
    us_anomaly_vix_change_pct: float = Field(default=4.0, gt=0.0)
    settlement_holding_days: int = Field(default=5, ge=1)
    # /pick wide funnel + DeepSeek ID filter (raw ScreenerCandidate objects are never mutated)
    pick_wide_funnel_top_n: int = Field(default=15, ge=8, le=20)
    pick_prefilter_keep_per_track: int = Field(default=8, ge=4, le=12)
    pick_search_symbols_per_track: int = Field(default=8, ge=4, le=12)
    # Long-term user preference memory (verbatim quote anchored, auto-expiring)
    user_preference_memory_enabled: bool = True
    preference_ttl_days: int = Field(default=14, ge=1, le=90)
    preference_max_notes_per_day: int = Field(default=3, ge=1, le=5)
    # Intraday event-driven sentinel (0-token scan; DeepSeek only decides whether to push)
    intraday_sentinel_enabled: bool = True
    intraday_poll_minutes: int = Field(default=30, ge=10, le=120)
    intraday_move_alert_pct: float = Field(default=3.5, gt=0.0)
    intraday_order_near_pct: float = Field(default=1.5, gt=0.0)


class UniverseSection(BaseModel):
    isa_long_term: list[str] = Field(default_factory=list)
    invest_tactical: list[str] = Field(default_factory=list)
    etf_symbols: list[str] = Field(default_factory=list)
    audition_enabled: bool = False
    audition_target_size: int = Field(default=150, ge=20, le=500)
    audition_sector_cap: int = Field(default=28, ge=5, le=200)
    audition_min_uk_eu_slots: int = Field(default=30, ge=0, le=150)
    audition_min_avg_dollar_vol_usd: float = Field(default=5_000_000.0, ge=0.0)
    audition_fundamentals_ttl_days: int = Field(default=21, ge=1, le=90)
    audition_info_concurrency: int = Field(default=2, ge=1, le=4)
    audition_info_delay_seconds: float = Field(default=0.45, ge=0.0, le=5.0)
    audition_indices: list[str] = Field(
        default_factory=lambda: ["SP500", "NASDAQ100", "FTSE100", "FTSE250", "EUROSTOXX50"]
    )
    audition_sync_online_indices: bool = True

    @field_validator("isa_long_term", "invest_tactical", "etf_symbols", "audition_indices", mode="before")
    @classmethod
    def _clean_universe_symbols(cls, v: Any) -> list[str]:
        return _normalize_symbol_list(v)


class AppConfig(BaseModel):
    env: EnvSettings
    app: AppSection = Field(default_factory=AppSection)
    models: ModelsSection = Field(default_factory=ModelsSection)
    schedule: ScheduleSection = Field(default_factory=ScheduleSection)
    strategy: StrategySection = Field(default_factory=StrategySection)
    watchlist: list[str] = Field(default_factory=list)
    universe: UniverseSection = Field(default_factory=UniverseSection)

    _yaml_path: Path | None = PrivateAttr(default=None)
    _yaml_mtime_ns: int | None = PrivateAttr(default=None)

    @field_validator("watchlist", mode="before")
    @classmethod
    def _clean_watchlist_symbols(cls, v: Any) -> list[str]:
        return _normalize_symbol_list(v)

    def reload_pools_if_changed(self) -> bool:
        """Hot-reload `watchlist` and `universe` in-place when config.yaml mtime changes."""
        if self._yaml_path is None:
            return False
        try:
            if not self._yaml_path.exists():
                return False
            current_mtime_ns = self._yaml_path.stat().st_mtime_ns
        except OSError as exc:
            logger.warning("Failed to stat config file %s: %s", self._yaml_path, exc)
            return False

        if self._yaml_mtime_ns is not None and current_mtime_ns == self._yaml_mtime_ns:
            return False

        try:
            with self._yaml_path.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            if not isinstance(yaml_data, dict):
                raise ValueError("config.yaml root must be a mapping")
            new_watchlist = _normalize_symbol_list(yaml_data.get("watchlist") or [])
            new_universe = UniverseSection(**(yaml_data.get("universe") or {}))
        except Exception as exc:
            # Record failed mtime so repeated calls don't spam logs until the file is edited again
            self._yaml_mtime_ns = current_mtime_ns
            logger.warning(
                "Hot-reload of %s failed (%s); keeping previous watchlist/universe.",
                self._yaml_path,
                exc,
            )
            return False

        self.watchlist = new_watchlist
        self.universe = new_universe
        self._yaml_mtime_ns = current_mtime_ns
        logger.info(
            "Hot-reloaded pools from %s: watchlist=%d, isa_universe=%d, invest_universe=%d",
            self._yaml_path,
            len(self.watchlist),
            len(self.universe.isa_long_term),
            len(self.universe.invest_tactical),
        )
        return True


def load_config(
    yaml_path: str | Path = "config.yaml",
    env_file: str | Path | None = ".env",
) -> AppConfig:
    """Load configuration from YAML file and .env / environment variables."""
    yaml_data: dict = {}
    p = Path(yaml_path)
    mtime_ns: int | None = None
    if p.exists():
        mtime_ns = p.stat().st_mtime_ns
        with p.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    if env_file is not None and Path(env_file).exists():
        env_settings = EnvSettings(_env_file=str(env_file))
    else:
        env_settings = EnvSettings(_env_file=None)

    cfg = AppConfig(
        env=env_settings,
        app=AppSection(**(yaml_data.get("app") or {})),
        models=ModelsSection(**(yaml_data.get("models") or {})),
        schedule=ScheduleSection(**(yaml_data.get("schedule") or {})),
        strategy=StrategySection(**(yaml_data.get("strategy") or {})),
        watchlist=list(yaml_data.get("watchlist") or []),
        universe=UniverseSection(**(yaml_data.get("universe") or {})),
    )
    cfg._yaml_path = p
    cfg._yaml_mtime_ns = mtime_ns
    return cfg
