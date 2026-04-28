"""Machine learning models for trading using technical indicators."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate common technical indicators."""

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        low_min = low.rolling(period).min()
        high_max = high.rolling(period).max()
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(3).mean()
        return k_percent, d_percent

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index (simplified)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        adx = 100 * di_diff / di_sum
        return adx


class MLTradingModel:
    """Machine learning model for trading."""

    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        """Initialize ML model.
        
        Args:
            model_type: 'logistic' for LogisticRegression or 'rf' for RandomForest
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

    def create_features(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Create technical indicator features.
        
        Args:
            df: DataFrame with OHLC data (must have 'Open', 'High', 'Low', 'Close')
            lookback: Lookback period for some indicators
            
        Returns:
            DataFrame with technical indicator features
        """
        features = pd.DataFrame(index=df.index)

        # Price features
        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        features["price_change"] = df["Close"].diff()

        # Moving averages
        features["sma_5"] = df["Close"].rolling(5).mean()
        features["sma_10"] = df["Close"].rolling(10).mean()
        features["sma_20"] = df["Close"].rolling(20).mean()
        features["ema_12"] = df["Close"].ewm(span=12).mean()

        # MA ratios
        features["ma_ratio_5_10"] = features["sma_5"] / features["sma_10"]
        features["ma_ratio_10_20"] = features["sma_10"] / features["sma_20"]

        # RSI
        features["rsi_14"] = TechnicalIndicators.rsi(df["Close"], 14)
        features["rsi_7"] = TechnicalIndicators.rsi(df["Close"], 7)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(df["Close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = hist

        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df["Close"])
        features["bb_upper"] = upper
        features["bb_middle"] = middle
        features["bb_lower"] = lower
        features["bb_width"] = upper - lower
        features["bb_position"] = (df["Close"] - lower) / (upper - lower)

        # Volatility
        features["volatility_20"] = df["Close"].rolling(20).std()
        features["volatility_ratio"] = features["volatility_20"] / features["volatility_20"].rolling(20).mean()

        # Volume-based (if available)
        if "Volume" in df.columns:
            features["volume_sma"] = df["Volume"].rolling(20).mean()
            features["volume_ratio"] = df["Volume"] / features["volume_sma"]

        # ADX
        if all(col in df.columns for col in ["High", "Low", "Close"]):
            features["adx"] = TechnicalIndicators.adx(df["High"], df["Low"], df["Close"])

        # Stochastic
        if all(col in df.columns for col in ["High", "Low", "Close"]):
            k, d = TechnicalIndicators.stochastic(df["High"], df["Low"], df["Close"])
            features["stoch_k"] = k
            features["stoch_d"] = d

        # Drop NaN rows
        features = features.dropna()
        self.feature_names = features.columns.tolist()

        return features

    def create_labels(self, df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.01) -> pd.Series:
        """Create target labels.
        
        Args:
            df: DataFrame with price data
            forward_periods: Periods ahead to look for signal
            threshold: Return threshold for buy/sell signals
            
        Returns:
            Series with labels (1: BUY, 0: HOLD, -1: SELL)
        """
        future_returns = df["Close"].pct_change(forward_periods).shift(-forward_periods)

        labels = pd.Series(0, index=future_returns.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1

        return labels[:-forward_periods]

    def prepare_data(
        self, df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.01, test_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and labels for training.
        
        Args:
            df: DataFrame with OHLC data
            forward_periods: Periods ahead for labels
            threshold: Return threshold
            test_split: Fraction of data for testing
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        features = self.create_features(df)
        labels = self.create_labels(df, forward_periods, threshold)

        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx].values
        y = labels.loc[common_idx].values

        # Split data
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=self.random_state)
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict signals.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels (-1, 0, 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature array
            
        Returns:
            Probability array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        if not hasattr(self.model, "predict_proba"):
            return None
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        # Precision/Recall per class
        metrics = {"accuracy": accuracy}
        for label in [-1, 0, 1]:
            mask = y_test == label
            if mask.sum() > 0:
                label_name = {-1: "SELL", 0: "HOLD", 1: "BUY"}[label]
                correct = np.mean(y_pred[mask] == label)
                metrics[f"{label_name}_accuracy"] = correct
                metrics[f"{label_name}_count"] = mask.sum()

        return metrics

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if model supports it.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            return None

        if hasattr(self.model, "coef_"):
            # Logistic Regression
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            # Random Forest
            importance = self.model.feature_importances_
        else:
            return None

        df = pd.DataFrame({"feature": self.feature_names, "importance": importance})
        return df.sort_values("importance", ascending=False)
