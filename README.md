# AI Trading Bot

A comprehensive AI trading bot project featuring machine learning agents, moving average crossover strategies, and paper trading simulation.

## Components

- `bot/ai_trading_bot.py` — core trading environment and ML agent implementation
- `bot/strategies.py` — trading strategy interfaces and MA crossover implementation
- `bot/paper_trading.py` — paper trading account with position tracking and P&L calculation
- `examples/run_bot.py` — ML agent training and backtesting workflow
- `examples/ma_crossover_backtest.py` — MA crossover strategy with paper trading
- `requirements.txt` — Python dependencies

## Features

- **MA Crossover Strategy**: Bullish/bearish signals based on fast and slow moving average crossovers
- **Paper Trading**: Full paper trading account with position management and commission tracking
- **Performance Metrics**: Sharpe ratio, win rate, P&L tracking, and equity history
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

3. Run ML agent example:
   ```powershell
   python .\examples\run_bot.py
   ```

## Example Output

The MA crossover backtest produces:
- Trade execution logs with entry/exit prices
- Performance statistics (total return, P&L, win rate, Sharpe ratio)
- Equity history snapshots
- Detailed trade log

## Configuration

Customize strategies in the example files:
- Adjust MA periods in `ma_crossover_backtest.py` (default: 10/20)
- Modify initial balance and commission rates
- Change data date ranges and symbols
