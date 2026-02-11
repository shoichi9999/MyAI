# Binance Algorithm Explorer

Binanceグローバルの取引ペアに対して、トレードアルゴリズムを自動探索・バックテストするシステム。

## 機能

- **1分足バックテスト** - Binance APIから取得した1分足データでリアルなシミュレーション
- **10種類の組み込み戦略** - SMA/EMAクロス、RSI、ボリンジャーバンド、MACD、ストキャスティクス等
- **AI自動探索** - Optunaベースのパラメータ最適化で全戦略×全シンボルを自動探索
- **Web UI** - リアルタイムの探索状況確認、手動バックテスト、結果ランキング
- **CLI対応** - Web UIなしでもコマンドラインから探索・バックテスト実行可能

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### Web UI (推奨)
```bash
python run.py
# → http://localhost:5000
```

### CLI - AI自動探索
```bash
python run.py --explore --symbol BTCUSDT --days 7 --trials 50
python run.py --explore --symbol ALL --days 14 --trials 100
```

### CLI - 単発バックテスト
```bash
python run.py --backtest --strategy SMA_Cross --symbol BTCUSDT --days 7
python run.py --backtest --strategy RSI_MeanReversion --symbol ETHUSDT --days 14
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

## プロジェクト構造

```
MyAI/
├── run.py                  # エントリーポイント
├── config/
│   └── settings.py         # 全体設定
├── data/
│   └── fetcher.py          # Binance APIデータ取得
├── backtest/
│   └── engine.py           # バックテストエンジン
├── strategies/
│   ├── base.py             # 戦略基底クラス＋インジケーター
│   └── builtin.py          # 組み込み戦略10種
├── explorer/
│   └── optimizer.py        # AI自動探索エンジン (Optuna)
├── webapp/
│   ├── app.py              # Flask Web UI
│   └── templates/
│       └── index.html      # フロントエンド
└── results/                # 探索結果JSON保存
```

## 免責事項

このソフトウェアは教育・研究目的で作成されています。暗号資産取引には高いリスクが伴います。すべての取引判断は自己責任で行ってください。
