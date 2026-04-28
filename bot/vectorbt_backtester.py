"""Vectorbt-based high-performance backtesting engine."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

logger = logging.getLogger(__name__)

ACTIONS = {-1: "SELL", 0: "HOLD", 1: "BUY"}


class VectorbtBacktester:
    """High-performance backtester using vectorbt library."""

    def __init__(self, initial_cash: float = 10000.0, fees: float = 0.001, freq: str = "D"):
        """Initialize vectorbt backtester.
        
        Args:
            initial_cash: Starting capital
            fees: Trading fees as percentage (0.1% = 0.001)
            freq: Frequency of data ('D' for daily, 'H' for hourly, etc.)
        """
        if vbt is None:
            raise ImportError("vectorbt is required. Install with: pip install vectorbt")

        self.initial_cash = initial_cash
        self.fees = fees
        self.freq = freq

    def backtest_ma_crossover(
        self,
        prices: pd.Series,
        fast_period: int = 10,
        slow_period: int = 20,
        entry_type: str = "Long",
    ) -> pd.DataFrame:
        """Backtest MA crossover strategy using vectorbt.
        
        Args:
            prices: Price series with datetime index
            fast_period: Fast MA period
            slow_period: Slow MA period
            entry_type: 'Long' for long-only, 'Short' for short-only, 'Both' for both
            
        Returns:
            DataFrame with backtest results
        """
        # Calculate moving averages
        fast_ma = prices.rolling(fast_period).mean()
        slow_ma = prices.rolling(slow_period).mean()

        # Generate signals
        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma

        # Create portfolio using vectorbt
        pf = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.fees,
            freq=self.freq,
        )

        return self._extract_results(pf, prices, fast_period, slow_period)

    def backtest_rsi_strategy(
        self, prices: pd.Series, period: int = 14, oversold: float = 30, overbought: float = 70
    ) -> pd.DataFrame:
        """Backtest RSI strategy using vectorbt.
        
        Args:
            prices: Price series with datetime index
            period: RSI period
            oversold: Oversold threshold for buy signals
            overbought: Overbought threshold for sell signals
            
        Returns:
            DataFrame with backtest results
        """
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        entries = rsi < oversold
        exits = rsi > overbought

        # Create portfolio
        pf = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.fees,
            freq=self.freq,
        )

        return self._extract_results(pf, prices, period, 0, strategy_name="RSI")

    def backtest_custom_signals(
        self, prices: pd.Series, entry_signals: pd.Series, exit_signals: pd.Series
    ) -> pd.DataFrame:
        """Backtest custom entry/exit signals using vectorbt.
        
        Args:
            prices: Price series with datetime index
            entry_signals: Boolean series indicating entry signals
            exit_signals: Boolean series indicating exit signals
            
        Returns:
            DataFrame with backtest results
        """
        pf = vbt.Portfolio.from_signals(
            close=prices,
            entries=entry_signals,
            exits=exit_signals,
            init_cash=self.initial_cash,
            fees=self.fees,
            freq=self.freq,
        )

        return self._extract_results(pf, prices, 0, 0, strategy_name="Custom")

    def _extract_results(
        self, pf: vbt.Portfolio, prices: pd.Series, param1: int = 0, param2: int = 0, strategy_name: str = "MA"
    ) -> pd.DataFrame:
        """Extract and format results from vectorbt portfolio.
        
        Args:
            pf: Portfolio object from vectorbt
            prices: Original price series
            param1: First strategy parameter
            param2: Second strategy parameter
            strategy_name: Name of the strategy
            
        Returns:
            DataFrame with formatted results
        """
        results = {
            "strategy": strategy_name,
            "parameters": f"{param1}/{param2}" if param2 > 0 else f"period={param1}",
            "start_date": prices.index[0],
            "end_date": prices.index[-1],
            "initial_cash": self.initial_cash,
            "final_value": float(pf.final_value()),
            "total_return_pct": float(pf.total_return() * 100),
            "annual_return_pct": float(pf.annualized_return() * 100),
            "annual_volatility_pct": float(pf.annualized_volatility() * 100),
            "sharpe_ratio": float(pf.sharpe_ratio()),
            "max_drawdown_pct": float(pf.max_drawdown() * 100),
            "win_rate": float(pf.trades.win_rate * 100),
            "trades": int(pf.trades.count()),
            "buy_and_hold_return_pct": float(((prices.iloc[-1] / prices.iloc[0]) - 1) * 100),
        }

        # Add trade statistics if any trades were made
        if pf.trades.count() > 0:
            results["avg_trade_pnl"] = float(pf.trades.pnl.mean())
            results["best_trade_pnl"] = float(pf.trades.pnl.max())
            results["worst_trade_pnl"] = float(pf.trades.pnl.min())
            results["recovery_factor"] = (
                float(pf.trades.pnl.sum() / abs(pf.trades.pnl.min()))
                if pf.trades.pnl.min() != 0
                else np.inf
            )
        else:
            results["avg_trade_pnl"] = 0.0
            results["best_trade_pnl"] = 0.0
            results["worst_trade_pnl"] = 0.0
            results["recovery_factor"] = 0.0

        return pd.DataFrame([results])

    def compare_strategies(self, prices: pd.Series, strategies: dict) -> pd.DataFrame:
        """Compare multiple strategies side-by-side.
        
        Args:
            prices: Price series with datetime index
            strategies: Dict of strategy configs
                {
                    'ma_10_20': {'type': 'ma', 'fast': 10, 'slow': 20},
                    'rsi_14': {'type': 'rsi', 'period': 14},
                    ...
                }
                
        Returns:
            DataFrame with comparison results
        """
        results = []

        for strategy_name, config in strategies.items():
            try:
                if config["type"] == "ma":
                    result = self.backtest_ma_crossover(prices, config["fast"], config["slow"])
                elif config["type"] == "rsi":
                    result = self.backtest_rsi_strategy(prices, config["period"])
                else:
                    logger.warning(f"Unknown strategy type: {config['type']}")
                    continue

                results.append(result)
            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}")

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def optimize_ma_crossover(
        self,
        prices: pd.Series,
        fast_range: range = range(5, 20, 2),
        slow_range: range = range(20, 50, 5),
    ) -> pd.DataFrame:
        """Optimize MA crossover parameters using grid search.
        
        Args:
            prices: Price series with datetime index
            fast_range: Range of fast MA periods to test
            slow_range: Range of slow MA periods to test
            
        Returns:
            DataFrame with optimization results sorted by return
        """
        results = []

        for fast in fast_range:
            for slow in slow_range:
                if fast >= slow:
                    continue

                try:
                    result = self.backtest_ma_crossover(prices, fast, slow)
                    results.append(result)
                except Exception as e:
                    logger.debug(f"Error testing {fast}/{slow}: {e}")

        if results:
            df = pd.concat(results, ignore_index=True)
            return df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
        else:
            return pd.DataFrame()
