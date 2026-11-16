"""Standalone CLI to manually run the 0-token multi-market universe audition."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from src.config import load_config
from src.data_providers.market_data import MarketDataService
from src.data_providers.t212_client import T212Client
from src.engine.screener import run_universe_audition
from src.storage.db import AdvisorDatabase


async def _run_cli(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.app.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    db = AdvisorDatabase(
        cfg.app.db_path,
        cycle_reset_hour=cfg.schedule.morning_report_hour,
        cycle_reset_minute=cfg.schedule.morning_report_minute,
    )
    await db.init()

    t212 = T212Client(
        isa_key=cfg.env.t212_isa_key,
        isa_secret=cfg.env.t212_isa_secret,
        invest_key=cfg.env.t212_invest_key,
        invest_secret=cfg.env.t212_invest_secret,
        base_url=cfg.env.t212_base_url,
        db=db,
    )
    market_data = MarketDataService(
        finnhub_key=cfg.env.finnhub_api_key,
        fred_key=cfg.env.fred_api_key,
        ttl_seconds=cfg.app.market_data_ttl_seconds,
        db=db,
    )
    try:
        res = await run_universe_audition(
            cfg,
            t212,
            market_data,
            db,
            sync_online_indices=not args.no_online_indices,
            sync_t212_metadata=not args.no_t212_sync,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    finally:
        await market_data.close()
        await t212.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manually audition and refresh the 150-stock /pick universe.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-online-indices", action="store_true", help="Use local seeds/SQLite index cache only")
    parser.add_argument("--no-t212-sync", action="store_true", help="Skip live T212 metadata refresh")
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
