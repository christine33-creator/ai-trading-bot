"""MA Crossover with Paper Trading example."""
import logging
from datetime import datetime

from bot import MarketDataLoader
from bot.strategies import MACrossoverStrategy
from bot.paper_trading import PaperTradingAccount

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ma_crossover_backtest(
    symbol: str = "SPY",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    fast_period: int = 10,
    slow_period: int = 20,
    initial_balance: float = 10000.0,
) -> None:
    """Run a complete MA crossover backtest with paper trading."""

    logger.info("Loading market data for %s", symbol)
    market_data = MarketDataLoader.load(symbol=symbol, start=start, end=end)

    if market_data.empty:
        logger.error("Failed to load market data")
        return

    logger.info("Market data loaded: %d days", len(market_data))

    strategy = MACrossoverStrategy(fast_period=fast_period, slow_period=slow_period)
    account = PaperTradingAccount(initial_balance=initial_balance)

    trade_log = []
    for idx in range(1, len(market_data)):
        current_data = market_data.iloc[:idx]
        current_row = market_data.iloc[idx]
        current_price = current_row["Close"]
        current_time = current_row.name

        signal = strategy.generate_signal(current_data)
        if signal:
            logger.info("[%s] Signal: %s at %.2f - %s", current_time.strftime("%Y-%m-%d"), signal.action, signal.price, signal.reason)

            if signal.action == 1 and account.get_position(symbol) == 0:
                qty = account.buy(symbol, current_price, current_time)
                trade_log.append({"date": current_time, "action": "BUY", "price": current_price, "quantity": qty})

            elif signal.action == -1 and account.get_position(symbol) > 0:
                qty = account.sell(symbol, current_price, current_time)
                trade_log.append({"date": current_time, "action": "SELL", "price": current_price, "quantity": qty})

        account.record_equity(current_time, current_price, symbol)

    if account.get_position(symbol) > 0:
        final_price = market_data.iloc[-1]["Close"]
        final_time = market_data.index[-1]
        account.sell(symbol, final_price, final_time)
        account.record_equity(final_time, final_price, symbol)

    stats = account.get_performance_stats()

    print("\n" + "=" * 60)
    print("MA CROSSOVER STRATEGY BACKTEST RESULTS")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Strategy: {fast_period}-day MA crossover {slow_period}-day MA")
    print(f"Period: {market_data.index[0].strftime('%Y-%m-%d')} to {market_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Initial Balance: ${stats['initial_balance']:.2f}")
    print(f"Final Equity: ${stats['final_equity']:.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Total P&L: ${stats['total_pnl']:.2f}")
    print(f"Trades Executed: {stats['trades_executed']}")
    print(f"Closed Trades: {stats['closed_trades']}")
    print(f"Winning Trades: {stats['winning_trades']}")
    print(f"Losing Trades: {stats['losing_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print("=" * 60)

    if trade_log:
        print("\nTrade Log (first 10 trades):")
        for trade in trade_log[:10]:
            print(f"  {trade['date'].strftime('%Y-%m-%d')} {trade['action']:4s} {trade['quantity']:3d} shares @ ${trade['price']:.2f}")

    print("\nEquity History (snapshots):")
    for entry in account.equity_history[::len(account.equity_history) // 5 or 1][:5]:
        print(f"  {entry['timestamp'].strftime('%Y-%m-%d')} Equity: ${entry['equity']:.2f} Cash: ${entry['cash']:.2f}")


if __name__ == "__main__":
    run_ma_crossover_backtest()
