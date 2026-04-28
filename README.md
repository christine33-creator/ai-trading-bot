# AI Trading Bot

A simple AI trading bot project that trains a machine learning agent to predict trades from historical price data and backtests strategy performance.

## Components

- `bot/ai_trading_bot.py` — core trading environment and agent implementation
- `examples/run_bot.py` — example training and backtesting workflow
- `requirements.txt` — Python dependencies

## Getting Started

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Run the example script:
   ```powershell
   python .\examples\run_bot.py
   ```

## Notes

- The example fetches price data using `yfinance` when available.
- If data cannot be retrieved, the bot falls back to a synthetic dataset.
- The agent uses scikit-learn for model training and simple feature-based predictions.
