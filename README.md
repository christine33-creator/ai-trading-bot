# AI Trading Bot

A comprehensive AI trading bot project featuring machine learning agents, moving average crossover strategies, and paper trading simulation.

## Components

- `bot/ai_trading_bot.py` — core trading environment and ML agent implementation
- `bot/strategies.py` — trading strategy interfaces and MA crossover implementation
- `bot/paper_trading.py` — paper trading account with position tracking and P&L calculation
- `bot/vectorbt_backtester.py` — high-performance backtesting with vectorbt library
- `bot/ml_models.py` — machine learning models using technical indicators
- `examples/run_bot.py` — ML agent training and backtesting workflow
- `examples/ma_crossover_backtest.py` — MA crossover strategy with paper trading
- `examples/vectorbt_examples.py` — vectorbt backtesting demonstrations
- `examples/ml_trading_models.py` — logistic regression and random forest examples
- `requirements.txt` — Python dependencies

## Features

- **MA Crossover Strategy**: Bullish/bearish signals based on fast and slow moving average crossovers
- **Paper Trading**: Full paper trading account with position management and commission tracking
- **Vectorbt Backtesting**: High-performance vectorized backtesting with multiple strategies
  - MA crossover backtest
  - RSI-based strategy
  - Custom signal backtesting
  - Strategy comparison
  - Parameter optimization via grid search
- **Machine Learning Models**: Technical indicator-based ML trading
  - Logistic Regression classifier
  - Random Forest classifier
  - Technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX
  - Feature engineering with 30+ technical indicators
  - Model evaluation and feature importance
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, P&L tracking
- **Market Data**: Real market data via `yfinance` with synthetic fallback
- **ML Agent**: Scikit-learn based agent for predictive trading

## Getting Started

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Run MA Crossover backtest:
   ```powershell
   python .\examples\ma_crossover_backtest.py
   ```

3. Run vectorbt backtesting examples:
   ```powershell
   python .\examples\vectorbt_examples.py
   ```

4. Run ML trading model examples:
   ```powershell
   python .\examples\ml_trading_models.py
   ```

5. Run ML agent example:
   ```powershell
   python .\examples\run_bot.py
   ```

## Machine Learning Models

Train technical indicator-based models for trading signals:

```python
from bot import MLTradingModel, MarketDataLoader

# Load data
market_data = MarketDataLoader.load(symbol="SPY")

# Create and train logistic regression model
model = MLTradingModel(model_type="logistic")
X_train, X_test, y_train, y_test = model.prepare_data(market_data, forward_periods=5)
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Get feature importance
importance = model.get_feature_importance()
print(importance.head(10))

# Predict signals (-1: SELL, 0: HOLD, 1: BUY)
predictions = model.predict(X_test)
```

### Technical Indicators

30+ automatically calculated indicators:
- **Momentum**: RSI, MACD, Stochastic, ADX
- **Trend**: Moving averages (SMA, EMA), ADX
- **Volatility**: Bollinger Bands, ATR, Volatility ratios
- **Volume**: Volume-weighted indicators

## Configuration

Customize strategies in the example files:
- Adjust MA periods in `ma_crossover_backtest.py` (default: 10/20)
- Modify initial balance and commission rates
- Change data date ranges and symbols
