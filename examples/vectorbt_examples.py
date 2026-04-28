"""Vectorbt backtesting examples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from bot import MarketDataLoader
from bot.vectorbt_backtester import VectorbtBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def example_ma_crossover():
    """Example: MA crossover backtest."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2022-01-01", end="2023-12-31")
    prices = market_data["Close"]

    logger.info("Creating backtester...")
    backtester = VectorbtBacktester(initial_cash=10000.0, fees=0.001)

    logger.info("Running MA 10/20 crossover backtest...")
    results = backtester.backtest_ma_crossover(prices, fast_period=10, slow_period=20)

    print("\n" + "=" * 70)
    print("MA CROSSOVER BACKTEST (Vectorbt)")
    print("=" * 70)
    for col in results.columns:
        value = results[col].iloc[0]
        if isinstance(value, float):
            print(f"{col:.<40} {value:.2f}")
        else:
            print(f"{col:.<40} {value}")
    print("=" * 70)


def example_compare_strategies():
    """Example: Compare multiple strategies."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2022-01-01", end="2023-12-31")
    prices = market_data["Close"]

    backtester = VectorbtBacktester(initial_cash=10000.0, fees=0.001)

    strategies = {
        "ma_5_10": {"type": "ma", "fast": 5, "slow": 10},
        "ma_10_20": {"type": "ma", "fast": 10, "slow": 20},
        "ma_10_30": {"type": "ma", "fast": 10, "slow": 30},
        "ma_20_50": {"type": "ma", "fast": 20, "slow": 50},
        "rsi_14": {"type": "rsi", "period": 14},
    }

    logger.info("Comparing %d strategies...", len(strategies))
    comparison = backtester.compare_strategies(prices, strategies)

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    print(comparison[["strategy", "parameters", "total_return_pct", "annual_return_pct", "sharpe_ratio", "max_drawdown_pct", "trades"]].to_string(index=False))
    print("=" * 100)


def example_optimize_parameters():
    """Example: Optimize MA crossover parameters."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2022-01-01", end="2023-12-31")
    prices = market_data["Close"]

    backtester = VectorbtBacktester(initial_cash=10000.0, fees=0.001)

    logger.info("Optimizing MA parameters (grid search)...")
    results = backtester.optimize_ma_crossover(prices, fast_range=range(5, 25, 2), slow_range=range(20, 60, 5))

    print("\n" + "=" * 110)
    print("MA CROSSOVER PARAMETER OPTIMIZATION (Top 10)")
    print("=" * 110)
    display_cols = ["parameters", "total_return_pct", "annual_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate", "trades"]
    print(results[display_cols].head(10).to_string(index=False))
    print("=" * 110)

    if len(results) > 0:
        best = results.iloc[0]
        print(f"\nBest Parameters: {best['parameters']}")
        print(f"Total Return: {best['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    print("Vectorbt Backtesting Examples\n")

    try:
        example_ma_crossover()
    except Exception as e:
        logger.error(f"MA crossover example failed: {e}")

    try:
        example_compare_strategies()
    except Exception as e:
        logger.error(f"Strategy comparison example failed: {e}")

    try:
        example_optimize_parameters()
    except Exception as e:
        logger.error(f"Parameter optimization example failed: {e}")
