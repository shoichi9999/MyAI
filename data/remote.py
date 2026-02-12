"""外部ストレージからCSVデータをダウンロードするモジュール

マニフェストファイル (data/manifest.json) にURLを定義し、
どのストレージサービスからでもデータを取得できる。

対応ストレージ:
  - 任意の公開URL (Cloudflare R2, S3, GCS, Dropbox, etc.)
  - GitHub Releases
  - 署名付きURL (S3 presigned, GCS signed)

manifest.json 例:
{
  "base_url": "https://your-bucket.r2.cloudflarestorage.com/klines",
  "files": {
    "BTCUSDT_7d.csv": {"size_mb": 12.3, "rows": 10080, "updated": "2025-01-15"},
    "ETHUSDT_7d.csv": {"size_mb": 11.8, "rows": 10080, "updated": "2025-01-15"}
  }
}
"""

import os
import json
import subprocess
import logging

logger = logging.getLogger(__name__)

MANIFEST_PATH = "data/manifest.json"
CSV_DIR = "data/csv"


def load_manifest() -> dict:
    """マニフェストファイルを読み込む"""
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save_manifest(manifest: dict):
    """マニフェストファイルを保存"""
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def download_file(url: str, dest: str, timeout: int = 300) -> bool:
    """URLからファイルをダウンロード (curl/wget)"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # curl を優先
    for attempt in range(4):
        try:
            result = subprocess.run(
                ["curl", "-fSL", "--max-time", str(timeout), "-o", dest, url],
                capture_output=True, text=True, timeout=timeout + 30,
            )
            if result.returncode == 0:
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                logger.info(f"Downloaded {dest} ({size_mb:.1f} MB)")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # wget フォールバック
        try:
            result = subprocess.run(
                ["wget", "-q", "--timeout", str(timeout), "-O", dest, url],
                capture_output=True, text=True, timeout=timeout + 30,
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        if attempt < 3:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Retry {attempt + 1}/3 in {wait}s...")
            import time
            time.sleep(wait)

    logger.error(f"Failed to download: {url}")
    return False


def sync_data(symbols: list = None, force: bool = False) -> dict:
    """マニフェストに基づいてCSVデータを同期する

    Args:
        symbols: 同期するシンボルのリスト (None=全て)
        force: 既存ファイルも再ダウンロード

    Returns:
        {"downloaded": [...], "skipped": [...], "failed": [...]}
    """
    manifest = load_manifest()
    if not manifest:
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        return {"downloaded": [], "skipped": [], "failed": []}

    base_url = manifest.get("base_url", "")
    files = manifest.get("files", {})

    result = {"downloaded": [], "skipped": [], "failed": []}

    for filename, meta in files.items():
        # シンボルフィルタ
        symbol = filename.split("_")[0]
        if symbols and symbol not in symbols:
            continue

        dest = os.path.join(CSV_DIR, filename)

        # 既存チェック
        if os.path.exists(dest) and not force:
            result["skipped"].append(filename)
            continue

        # URLを構築 (ファイルごとの個別URLも対応)
        url = meta.get("url") or f"{base_url}/{filename}"

        logger.info(f"Downloading {filename}...")
        if download_file(url, dest):
            result["downloaded"].append(filename)
        else:
            result["failed"].append(filename)

    return result


def list_remote_symbols() -> list:
    """マニフェストからリモートで利用可能なシンボル一覧を取得"""
    manifest = load_manifest()
    files = manifest.get("files", {})
    symbols = set()
    for filename in files:
        symbol = filename.split("_")[0]
        symbols.add(symbol)
    return sorted(symbols)
