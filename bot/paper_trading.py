"""Paper trading account and execution engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ACTIONS = {-1: "SELL", 0: "HOLD", 1: "BUY"}


@dataclass
class Trade:
    """Represents a single executed trade."""

    entry_time: datetime
    entry_price: float
    quantity: int
    side: str  # "BUY" or "SELL"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    commission: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None

    def close(self, exit_price: float, exit_time: datetime, commission: float = 0.0) -> None:
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.commission = commission
        self.pnl = (exit_price - self.entry_price) * self.quantity - commission
        if self.entry_price > 0:
            self.return_pct = ((exit_price - self.entry_price) / self.entry_price) * 100


@dataclass
class PaperTradingAccount:
    """Paper trading account with position tracking and P&L calculation."""

    initial_balance: float
    commission_rate: float = 0.001  # 0.1% per trade
    balance: float = field(init=False)
    positions: dict[str, int] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    equity_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.balance = self.initial_balance

    def get_position(self, symbol: str = "SPY") -> int:
        """Get current position quantity."""
        return self.positions.get(symbol, 0)

    def get_equity(self, current_price: float, symbol: str = "SPY") -> float:
        """Calculate total equity (cash + position value)."""
        position_value = self.get_position(symbol) * current_price
        return self.balance + position_value

    def buy(self, symbol: str, price: float, timestamp: datetime, max_shares: Optional[int] = None) -> int:
        """Execute a buy order with available cash."""
        commission = price * self.commission_rate
        total_cost_per_share = price + commission
        available_cash = self.balance
        max_purchasable = int(available_cash / total_cost_per_share)

        if max_purchasable <= 0:
            logger.warning("Insufficient cash to buy. Available: %.2f", available_cash)
            return 0

        quantity = max_shares if max_shares and max_shares <= max_purchasable else max_purchasable
        total_cost = quantity * (price + commission)

        self.balance -= total_cost
        current_pos = self.get_position(symbol)
        self.positions[symbol] = current_pos + quantity

        trade = Trade(entry_time=timestamp, entry_price=price, quantity=quantity, side="BUY", commission=commission * quantity)
        self.trades.append(trade)

        logger.info("BUY: %d shares at %.2f (total cost: %.2f)", quantity, price, total_cost)
        return quantity

    def sell(self, symbol: str, price: float, timestamp: datetime, quantity: Optional[int] = None) -> int:
        """Execute a sell order."""
        current_pos = self.get_position(symbol)
        if current_pos <= 0:
            logger.warning("No position to sell")
            return 0

        sell_qty = quantity if quantity and quantity <= current_pos else current_pos
        commission = sell_qty * price * self.commission_rate
        proceeds = sell_qty * price - commission

        self.balance += proceeds
        self.positions[symbol] = current_pos - sell_qty

        if self.trades:
            last_trade = self.trades[-1]
            if last_trade.side == "BUY" and last_trade.exit_time is None:
                last_trade.close(price, timestamp, commission)

        logger.info("SELL: %d shares at %.2f (proceeds: %.2f)", sell_qty, price, proceeds)
        return sell_qty

    def record_equity(self, timestamp: datetime, current_price: float, symbol: str = "SPY") -> None:
        """Record equity snapshot for performance tracking."""
        equity = self.get_equity(current_price, symbol)
        self.equity_history.append({"timestamp": timestamp, "equity": equity, "cash": self.balance, "position": self.get_position(symbol)})

    def get_performance_stats(self) -> dict:
        """Calculate performance metrics."""
        if not self.equity_history:
            return {}

        initial_equity = self.initial_balance
        final_equity = self.equity_history[-1]["equity"]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100

        closed_trades = [t for t in self.trades if t.pnl is not None]
        total_pnl = sum(t.pnl for t in closed_trades)
        win_count = len([t for t in closed_trades if t.pnl > 0])
        loss_count = len([t for t in closed_trades if t.pnl < 0])

        equity_values = [h["equity"] for h in self.equity_history]
        if len(equity_values) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * (252**0.5) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "initial_balance": initial_equity,
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "total_pnl": total_pnl,
            "trades_executed": len(self.trades),
            "closed_trades": len(closed_trades),
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": (win_count / len(closed_trades) * 100) if closed_trades else 0.0,
            "sharpe_ratio": sharpe_ratio,
        }
