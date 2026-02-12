"""開発環境でマニフェストに基づいてCSVデータをダウンロードするスクリプト

Usage:
    # 全シンボルをダウンロード
    python sync_data.py

    # 特定シンボルのみ
    python sync_data.py --symbol BTCUSDT ETHUSDT

    # 強制再ダウンロード
    python sync_data.py --force
"""

import argparse
import logging
from data.remote import sync_data, load_manifest, list_remote_symbols

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sync CSV data from remote storage")
    parser.add_argument("--symbol", nargs="+", help="Specific symbols to sync")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    manifest = load_manifest()
    if not manifest:
        logger.error("data/manifest.json not found.")
        logger.error("Run upload_data.py locally first to create the manifest.")
        return

    remote_symbols = list_remote_symbols()
    logger.info(f"=== Data Sync ===")
    logger.info(f"Provider: {manifest.get('provider', 'unknown')}")
    logger.info(f"Available: {len(remote_symbols)} symbols: {', '.join(remote_symbols)}")
    logger.info(f"Last upload: {manifest.get('last_upload', 'unknown')}")
    logger.info("")

    result = sync_data(symbols=args.symbol, force=args.force)

    if result["downloaded"]:
        logger.info(f"Downloaded: {len(result['downloaded'])} files")
        for f in result["downloaded"]:
            logger.info(f"  + {f}")
    if result["skipped"]:
        logger.info(f"Skipped (already exists): {len(result['skipped'])} files")
    if result["failed"]:
        logger.error(f"Failed: {len(result['failed'])} files")
        for f in result["failed"]:
            logger.error(f"  ! {f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
