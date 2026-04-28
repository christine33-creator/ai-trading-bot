"""Trading strategy implementations."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal with metadata."""

    timestamp: pd.Timestamp
    action: int  # -1: SELL, 0: HOLD, 1: BUY
    price: float
    reason: str
    confidence: float = 1.0


class Strategy(ABC):
    """Base strategy interface."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Generate a trading signal based on the latest data."""
        pass

    @abstractmethod
    def on_data(self, df: pd.DataFrame) -> None:
        """Process new market data."""
        pass


class MACrossoverStrategy(Strategy):
    """Moving Average Crossover Strategy.
    
    Generates buy signals when fast MA crosses above slow MA (bullish).
    Generates sell signals when fast MA crosses below slow MA (bearish).
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.last_signal: Optional[int] = None

    def on_data(self, df: pd.DataFrame) -> None:
        """Process new market data and cache state."""
        pass

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Generate signal based on MA crossover."""
        if len(df) < self.slow_period + 1:
            return None

        prices = df["Close"]
        fast_ma = prices.rolling(self.fast_period).mean()
        slow_ma = prices.rolling(self.slow_period).mean()

        current_price = prices.iloc[-1]
        current_timestamp = prices.index[-1]

        fast_curr = fast_ma.iloc[-1]
        slow_curr = slow_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_prev = slow_ma.iloc[-2]

        if pd.isna(fast_curr) or pd.isna(slow_curr) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return None

        # Bullish crossover: fast MA crosses above slow MA
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            if self.last_signal != 1:
                self.last_signal = 1
                return Signal(
                    timestamp=current_timestamp,
                    action=1,
                    price=current_price,
                    reason=f"Bullish crossover: {self.fast_period}MA ({fast_curr:.2f}) crossed above {self.slow_period}MA ({slow_curr:.2f})",
                    confidence=0.8,
                )

        # Bearish crossover: fast MA crosses below slow MA
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            if self.last_signal != -1:
                self.last_signal = -1
                return Signal(
                    timestamp=current_timestamp,
                    action=-1,
                    price=current_price,
                    reason=f"Bearish crossover: {self.fast_period}MA ({fast_curr:.2f}) crossed below {self.slow_period}MA ({slow_curr:.2f})",
                    confidence=0.8,
                )

        return None
