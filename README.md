# Binance Algorithm Explorer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Binanceグローバルの取引ペアに対して、トレードアルゴリズムを**自動探索・バックテスト**するシステム。
Optunaによるパラメータ最適化と戦略構造の自動生成で、有望な売買ロジックを効率的に発見します。

## 主な機能

| 機能 | 概要 |
|------|------|
| **1分足バックテスト** | Binance APIから取得した1分足データでリアルなシミュレーション |
| **10種類の組み込み戦略** | SMA/EMAクロス、RSI、ボリンジャーバンド、MACD、ストキャスティクス等 |
| **戦略自動生成** | 14種類の条件ルールを組み合わせ、戦略の構造自体をOptunaで自動探索 |
| **AI自動探索** | Optunaベースのパラメータ最適化で全戦略×全シンボルを自動探索 |
| **データパイプライン** | Binance API直接取得、CSV保存、クラウドストレージ(R2/S3/GCS)連携 |
| **Web UI** | リアルタイムの探索状況確認、手動バックテスト、結果ランキング |
| **CLI対応** | Web UIなしでもコマンドラインから探索・バックテスト実行可能 |

## クイックスタート

```bash
# 1. 依存関係インストール
pip install -r requirements.txt

# 2. データ取得（例: BTC 7日分）
python fetch_data.py --symbol BTCUSDT --days 7

# 3. Web UI起動
python run.py
# → http://localhost:5000 でブラウザからアクセス
```

## 使い方

### Web UI (推奨)
```bash
python run.py
# → http://localhost:5000
```

### CLI - AI自動探索（固定戦略のパラメータ最適化）
```bash
python run.py --explore --symbol BTCUSDT --days 7 --trials 50
python run.py --explore --symbol ALL --days 14 --trials 100
```

### CLI - 戦略自動生成（構造自体の探索）
```bash
python run.py --generate --symbol BTCUSDT --days 7 --trials 500
```

### CLI - 単発バックテスト
```bash
python run.py --backtest --strategy SMA_Cross --symbol BTCUSDT --days 7
python run.py --backtest --strategy RSI_MeanReversion --symbol ETHUSDT --days 14
```

## データ取得

データは以下の優先順位で取得されます:

1. `data/csv/` のローカルCSVファイル（最優先）
2. リモートストレージ（`data/manifest.json` 経由）
3. `data/cache/` のParquetキャッシュ（1時間以内）
4. Binance APIからライブ取得（フォールバック）

### Binance APIから直接取得
```bash
# 単一シンボル、7日分
python fetch_data.py --symbol BTCUSDT --days 7

# 複数シンボル
python fetch_data.py --symbol BTCUSDT ETHUSDT SOLUSDT --days 14

# デフォルト15ペア
python fetch_data.py --days 7
```

### リモートストレージから同期
```bash
# 全シンボルをダウンロード
python sync_data.py

# 特定シンボルのみ
python sync_data.py --symbol BTCUSDT ETHUSDT

# 強制再ダウンロード
python sync_data.py --force
```

### クラウドストレージへアップロード
```bash
# Cloudflare R2（推奨・エグレス無料）
python upload_data.py --provider r2 --bucket my-klines --account-id YOUR_CF_ACCOUNT_ID

# AWS S3
python upload_data.py --provider s3 --bucket my-klines-bucket
```

## 組み込み戦略

| 戦略 | タイプ | 説明 |
|------|--------|------|
| SMA_Cross | トレンドフォロー | 単純移動平均のゴールデン/デッドクロス |
| EMA_Cross | トレンドフォロー | 指数移動平均のクロスオーバー |
| RSI_MeanReversion | 平均回帰 | RSIの買われすぎ/売られすぎで逆張り |
| BollingerBand_Breakout | ブレイクアウト | ボリンジャーバンド突破で順張り |
| BollingerBand_MeanReversion | 平均回帰 | ボリンジャーバンド到達で逆張り |
| MACD_Strategy | トレンドフォロー | MACDラインのクロスオーバー |
| Stochastic_Strategy | オシレーター | ストキャスティクスのK/Dクロス |
| Triple_EMA | トレンドフォロー | 3本のEMAの配列で判断 |
| RSI_MACD_Combo | コンボ | RSI + MACDの複合条件 |
| ATR_Breakout | ブレイクアウト | ATRベースのボラティリティブレイク |

## 戦略自動生成（CompositeStrategy）

`--generate` モードでは、以下の14種類の条件ルールを動的に組み合わせて新しい戦略を自動生成します:

| 条件タイプ | 説明 |
|-----------|------|
| rsi_threshold | RSI閾値による判定 |
| ema_cross | EMAクロスオーバー |
| sma_cross | SMAクロスオーバー |
| bb_position | ボリンジャーバンド位置 |
| macd_hist_sign | MACDヒストグラム符号 |
| macd_cross | MACDクロスオーバー |
| stoch_threshold | ストキャスティクス閾値 |
| stoch_cross | ストキャスティクスクロス |
| atr_breakout | ATRブレイクアウト |
| price_vs_sma | 価格とSMAの位置関係 |
| price_vs_ema | 価格とEMAの位置関係 |
| volume_spike | 出来高スパイク |
| price_momentum | 価格モメンタム |
| candle_body | ローソク足実体 |

1〜4個の条件をAND/ORロジックで組み合わせ、買い・売りそれぞれの条件を自動探索します。

## プロジェクト構造

```
MyAI/
├── run.py                  # エントリーポイント (Web UI / CLI)
├── fetch_data.py           # Binance APIからCSVデータ取得
├── sync_data.py            # リモートストレージからデータ同期
├── upload_data.py          # クラウドストレージへデータアップロード
├── config/
│   └── settings.py         # 全体設定（デフォルトシンボル、バックテスト条件等）
├── data/
│   ├── fetcher.py          # Binance APIデータ取得 + キャッシュ管理
│   ├── remote.py           # リモートストレージ連携 (R2/S3/GCS)
│   ├── csv/                # CSVデータ保存先
│   └── cache/              # Parquetキャッシュ保存先
├── backtest/
│   └── engine.py           # バックテストエンジン
├── strategies/
│   ├── base.py             # 戦略基底クラス＋インジケーター
│   ├── builtin.py          # 組み込み戦略10種
│   └── composer.py         # 動的戦略生成 (CompositeStrategy)
├── explorer/
│   ├── optimizer.py        # AI自動探索エンジン (Optuna)
│   └── generator.py        # 戦略自動生成エンジン
├── webapp/
│   ├── app.py              # Flask Web UI
│   └── templates/
│       └── index.html      # フロントエンド
└── results/                # 探索結果JSON保存
```

## 技術スタック

- **Python 3.8+** - メイン言語
- **Optuna** - ハイパーパラメータ最適化
- **pandas / NumPy** - データ処理・数値計算
- **Flask** - Web UI
- **Binance API** - マーケットデータ取得

## 免責事項

このソフトウェアは教育・研究目的で作成されています。実際の暗号資産取引には高いリスクが伴います。すべての取引判断は自己責任で行ってください。
