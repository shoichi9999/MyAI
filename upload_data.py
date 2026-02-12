"""ローカル環境でCSVデータを外部ストレージにアップロードし、マニフェストを更新するスクリプト

対応サービス:
  - Cloudflare R2 (S3互換、無料エグレス)  ← おすすめ
  - AWS S3
  - GCS

Usage:
    # Cloudflare R2にアップロード
    python upload_data.py --provider r2 \
        --bucket my-klines \
        --account-id YOUR_CF_ACCOUNT_ID

    # AWS S3にアップロード
    python upload_data.py --provider s3 --bucket my-klines-bucket

    # 特定シンボルのみ
    python upload_data.py --provider r2 --bucket my-klines --symbol BTCUSDT ETHUSDT

事前準備:
    pip install boto3  # S3/R2共通
    # R2の場合: Cloudflareダッシュボード → R2 → APIトークン作成
    # 環境変数にセット:
    #   AWS_ACCESS_KEY_ID=xxx
    #   AWS_SECRET_ACCESS_KEY=xxx
"""

import os
import sys
import json
import argparse
from datetime import datetime

CSV_DIR = "data/csv"
MANIFEST_PATH = "data/manifest.json"


def get_s3_client(provider, account_id=None):
    """S3互換クライアントを作成"""
    import boto3

    if provider == "r2":
        if not account_id:
            raise ValueError("R2 requires --account-id")
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        return boto3.client("s3", endpoint_url=endpoint)
    elif provider == "s3":
        return boto3.client("s3")
    elif provider == "gcs":
        return boto3.client("s3", endpoint_url="https://storage.googleapis.com")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_public_url(provider, bucket, key, account_id=None):
    """公開URLを生成"""
    if provider == "r2":
        # R2のパブリックアクセスURL (カスタムドメインまたはr2.dev)
        return f"https://pub-{account_id}.r2.dev/{key}"
    elif provider == "s3":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    elif provider == "gcs":
        return f"https://storage.googleapis.com/{bucket}/{key}"
    return ""


def upload_files(provider, bucket, symbols=None, account_id=None, prefix="klines"):
    """CSVファイルをアップロードしマニフェストを更新"""
    client = get_s3_client(provider, account_id)

    # 既存マニフェスト読み込み
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    manifest.setdefault("files", {})

    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv") and not f.startswith("_")]

    if symbols:
        csv_files = [f for f in csv_files if f.split("_")[0] in symbols]

    if not csv_files:
        print("No CSV files to upload")
        return

    print(f"Uploading {len(csv_files)} files to {provider}://{bucket}/{prefix}/")

    for filename in sorted(csv_files):
        filepath = os.path.join(CSV_DIR, filename)
        key = f"{prefix}/{filename}"
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        print(f"  Uploading {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)
        try:
            client.upload_file(filepath, bucket, key)
            print("OK")

            # マニフェスト更新
            url = get_public_url(provider, bucket, key, account_id)
            # 行数を取得
            with open(filepath) as f:
                rows = sum(1 for _ in f) - 1  # ヘッダー除く

            manifest["files"][filename] = {
                "url": url,
                "size_mb": round(size_mb, 1),
                "rows": rows,
                "updated": datetime.utcnow().strftime("%Y-%m-%d"),
            }
        except Exception as e:
            print(f"FAILED: {e}")

    # base_urlを設定
    if manifest["files"]:
        first_file = next(iter(manifest["files"].values()))
        if "url" in first_file:
            # URLからbase_urlを逆算
            url = first_file["url"]
            base_url = url.rsplit("/", 1)[0]
            manifest["base_url"] = base_url

    manifest["provider"] = provider
    manifest["bucket"] = bucket
    manifest["last_upload"] = datetime.utcnow().isoformat()

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest updated: {MANIFEST_PATH}")
    print("Next: git add data/manifest.json && git commit && git push")


def main():
    parser = argparse.ArgumentParser(description="Upload CSV data to cloud storage")
    parser.add_argument("--provider", required=True, choices=["r2", "s3", "gcs"])
    parser.add_argument("--bucket", required=True, help="Bucket name")
    parser.add_argument("--account-id", help="Cloudflare Account ID (R2 only)")
    parser.add_argument("--symbol", nargs="+", help="Specific symbols to upload")
    parser.add_argument("--prefix", default="klines", help="Object key prefix")
    args = parser.parse_args()

    upload_files(
        provider=args.provider,
        bucket=args.bucket,
        symbols=args.symbol,
        account_id=args.account_id,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
