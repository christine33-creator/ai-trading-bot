# AI Trading Bot

A comprehensive AI trading bot project featuring machine learning agents, moving average crossover strategies, and paper trading simulation.

## Components

- `bot/ai_trading_bot.py` — core trading environment and ML agent implementation
- `bot/strategies.py` — trading strategy interfaces and MA crossover implementation
- `bot/paper_trading.py` — paper trading account with position tracking and P&L calculation
- `bot/vectorbt_backtester.py` — high-performance backtesting with vectorbt library
- `examples/run_bot.py` — ML agent training and backtesting workflow
- `examples/ma_crossover_backtest.py` — MA crossover strategy with paper trading
- `examples/vectorbt_examples.py` — vectorbt backtesting demonstrations
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

4. Run ML agent example:
   ```powershell
   python .\examples\run_bot.py
   ```

## Vectorbt Backtesting

Vectorbt enables high-performance backtesting with vectorized operations:

```python
from bot import VectorbtBacktester, MarketDataLoader

# Load data
prices = MarketDataLoader.load(symbol="SPY")["Close"]

# Create backtester
backtester = VectorbtBacktester(initial_cash=10000, fees=0.001)

# Backtest strategy
results = backtester.backtest_ma_crossover(prices, fast_period=10, slow_period=20)

# Optimize parameters
optimization = backtester.optimize_ma_crossover(prices)

# Compare strategies
comparison = backtester.compare_strategies(prices, {
    'ma_10_20': {'type': 'ma', 'fast': 10, 'slow': 20},
    'rsi_14': {'type': 'rsi', 'period': 14},
})
```

## Configuration

Customize strategies in the example files:
- Adjust MA periods in `ma_crossover_backtest.py` (default: 10/20)
- Modify initial balance and commission rates
- Change data date ranges and symbols
