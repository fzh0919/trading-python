"""Async SQLite repository for portfolio snapshots, reports, picks, settlements, and daily chat history."""

from __future__ import annotations

import logging

from src.storage.db.core import (
    LONDON_TZ,
    DatabaseCore,
    _ensure_columns,
    get_9am_cycle_id,
    get_trading_cycle_id,
)
from src.storage.db.market_cache_repo import MarketCacheRepoMixin
from src.storage.db.memory_and_wallet_repo import MemoryAndWalletRepoMixin
from src.storage.db.ops_and_chat_repo import OpsAndChatRepoMixin
from src.storage.db.portfolio_repo import PortfolioRepoMixin
from src.storage.db.recommendations_repo import (
    _RECOMMENDATION_SELECT_COLS,
    RecommendationsRepoMixin,
    _row_to_recommendation,
)
from src.storage.db.schema import SchemaMixin

logger = logging.getLogger(__name__)


class AdvisorDatabase(
    SchemaMixin,
    PortfolioRepoMixin,
    RecommendationsRepoMixin,
    OpsAndChatRepoMixin,
    MarketCacheRepoMixin,
    MemoryAndWalletRepoMixin,
    DatabaseCore,
):
    """Unified async SQLite repository sharing a single write lock via DatabaseCore._connect."""

    pass


__all__ = [
    "AdvisorDatabase",
    "LONDON_TZ",
    "get_9am_cycle_id",
    "get_trading_cycle_id",
]
