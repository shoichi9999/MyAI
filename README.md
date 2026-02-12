# Bitcoin Backtest

ビットコインの各種トレード戦略をバックテストするCLIツール

## 特徴

- **複数の戦略に対応** - 移動平均、RSI、MACD、ボリンジャーバンド等
- **Yahoo Finance / Binance からデータ取得** - キャッシュ機能付き
- **戦略比較** - 複数戦略の一括比較分析
- **パラメータ最適化** - グリッドサーチによる最適パラメータ探索

## 利用可能な戦略

| 戦略名 | 説明 |
|--------|------|
| sma_crossover | シンプル移動平均クロスオーバー |
| ema_crossover | 指数移動平均クロスオーバー |
| triple_sma | トリプル移動平均 |
| rsi | RSI（相対力指数） |
| macd | MACD |
| momentum | モメンタム |
| bollinger_bands | ボリンジャーバンド |
| mean_reversion | 平均回帰 |

## インストール & 実行

1. **リポジトリをクローン**
```bash
git clone https://github.com/shoichi9999/MyAI.git
cd MyAI
```

2. **依存関係をインストール**
```bash
pip install -r requirements.txt
```

3. **デモを実行**
```bash
python main.py
```

## 必要要件

- Python 3.8+
- yfinance

pandas, numpy 等は Anaconda 環境であれば同梱されています。

## 免責事項

このソフトウェアは教育・研究目的で作成されています。
- **投資リスク**: 暗号資産取引には高いリスクが伴います
- **自己責任**: すべての取引判断は自己責任で行ってください

## ライセンス

MIT License
