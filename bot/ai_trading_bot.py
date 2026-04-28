"""AI agent trading bot implementation."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ACTIONS = {-1: "SELL", 0: "HOLD", 1: "BUY"}


def _simulate_price_series(length: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0005, scale=0.02, size=length)
    prices = 100 * np.exp(np.cumsum(steps))
    df = pd.DataFrame({"Close": prices})
    df.index = pd.date_range(end=pd.Timestamp.today(), periods=length, freq="B")
    return df


class MarketDataLoader:
    @staticmethod
    def load(symbol: str = "SPY", start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance is not installed; using synthetic data instead.")
            return _simulate_price_series()

        try:
            logger.info("Fetching historical data for %s", symbol)
            ticker = yf.Ticker(symbol)
            frame = ticker.history(start=start, end=end)
            if frame.empty:
                logger.warning("No data returned for %s; using synthetic data instead.", symbol)
                return _simulate_price_series()
            return frame[["Close"]].copy()
        except Exception as exc:
            logger.warning("Failed to load market data: %s. Using synthetic data.", exc)
            return _simulate_price_series()


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    prices = df["Close"].copy()
    returns = prices.pct_change()
    df_feat = pd.DataFrame(index=prices.index)
    df_feat["close"] = prices
    df_feat["return_1d"] = returns
    df_feat["ma_5"] = prices.rolling(5).mean()
    df_feat["ma_10"] = prices.rolling(10).mean()
    df_feat["std_10"] = prices.rolling(10).std()
    df_feat["momentum_5"] = prices / prices.shift(5) - 1
    df_feat["ma_ratio"] = df_feat["ma_5"] / df_feat["ma_10"]
    df_feat = df_feat.dropna()
    return df_feat


def create_labels(features: pd.DataFrame, threshold: float = 0.005) -> pd.Series:
    future_returns = features["close"].pct_change().shift(-1)
    labels = pd.Series(0, index=features.index)
    labels[future_returns > threshold] = 1
    labels[future_returns < -threshold] = -1
    return labels[:-1]


class TradingAgent:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="multinomial"))

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have the same number of samples.")
        logger.info("Training agent on %d examples", len(labels))
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

    def action_names(self, actions: np.ndarray) -> list[str]:
        return [ACTIONS.get(int(a), "HOLD") for a in actions]


class TradingEnvironment:
    def __init__(self, initial_cash: float = 10000.0, transaction_cost: float = 0.001):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

    def backtest(self, prices: pd.Series, actions: np.ndarray) -> pd.DataFrame:
        if len(prices) != len(actions):
            raise ValueError("Prices and actions must have the same length.")

        cash = self.initial_cash
        shares = 0
        history = []
        for date, price, action in zip(prices.index, prices.values, actions):
            if action == 1 and cash >= price:
                quantity = int(cash // price)
                cost = quantity * price * (1 + self.transaction_cost)
                cash -= cost
                shares += quantity
            elif action == -1 and shares > 0:
                proceeds = shares * price * (1 - self.transaction_cost)
                cash += proceeds
                shares = 0
            equity = cash + shares * price
            history.append({"date": date, "price": price, "action": ACTIONS.get(int(action), "HOLD"), "cash": cash, "shares": shares, "equity": equity})

        result = pd.DataFrame(history).set_index("date")
        result["return"] = result["equity"].pct_change().fillna(0.0)
        return result

    def evaluate(self, backtest_df: pd.DataFrame) -> dict[str, float]:
        returns = backtest_df["return"]
        total_return = backtest_df["equity"].iloc[-1] / self.initial_cash - 1.0
        annualized = ((1 + total_return) ** (252 / len(returns))) - 1.0 if len(returns) > 0 else 0.0
        return {
            "starting_cash": self.initial_cash,
            "ending_equity": float(backtest_df["equity"].iloc[-1]),
            "total_return": float(total_return),
            "annualized_return": float(annualized),
            "trade_count": int((backtest_df["action"] != "HOLD").sum()),
        }
