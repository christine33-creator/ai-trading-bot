"""Simple MA Crossover test without package installation."""
import sys
from pathlib import Path

# Add bot to path
bot_path = Path(__file__).parent.parent / "bot"
sys.path.insert(0, str(bot_path.parent))

import logging
from datetime import datetime

from bot import MarketDataLoader
from bot.strategies import MACrossoverStrategy
from bot.paper_trading import PaperTradingAccount

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_quick_test():
    """Quick test of MA crossover with paper trading."""
    logger.info("Starting MA Crossover quick test...")

    # Load market data
    market_data = MarketDataLoader.load(symbol="SPY", start="2023-06-01", end="2023-12-31")
    logger.info("Loaded %d days of market data", len(market_data))

    # Create strategy and account
    strategy = MACrossoverStrategy(fast_period=10, slow_period=20)
    account = PaperTradingAccount(initial_balance=10000.0)

    trades_made = 0

    # Process each day
    for idx in range(1, min(len(market_data), 100)):  # Limit to first 100 days for quick test
        current_data = market_data.iloc[:idx]
        current_row = market_data.iloc[idx]
        current_price = current_row["Close"]
        current_time = current_row.name

        signal = strategy.generate_signal(current_data)
        if signal:
            logger.info("[%s] Signal: Action=%d at Price=%.2f - %s", 
                       current_time.strftime("%Y-%m-%d"), signal.action, signal.price, signal.reason)

            if signal.action == 1 and account.get_position("SPY") == 0:
                account.buy("SPY", current_price, current_time)
                trades_made += 1

            elif signal.action == -1 and account.get_position("SPY") > 0:
                account.sell("SPY", current_price, current_time)
                trades_made += 1

        account.record_equity(current_time, current_price, "SPY")

    # Final sell if holding
    if account.get_position("SPY") > 0:
        final_price = market_data.iloc[-1]["Close"]
        final_time = market_data.index[-1]
        account.sell("SPY", final_price, final_time)

    # Print results
    stats = account.get_performance_stats()
    print("\n" + "="*60)
    print("MA CROSSOVER QUICK TEST RESULTS")
    print("="*60)
    print(f"Initial Balance: ${stats['initial_balance']:.2f}")
    print(f"Final Equity: ${stats['final_equity']:.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Total P&L: ${stats['total_pnl']:.2f}")
    print(f"Trades Executed: {trades_made}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print("="*60)
    logger.info("Test complete!")


if __name__ == "__main__":
    run_quick_test()
