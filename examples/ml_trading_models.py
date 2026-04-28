"""Machine learning trading model examples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from bot import MarketDataLoader
from bot.ml_models import MLTradingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def example_logistic_regression():
    """Example: Logistic Regression model."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2021-01-01", end="2023-12-31")

    logger.info("Creating and training logistic regression model...")
    model = MLTradingModel(model_type="logistic")

    X_train, X_test, y_train, y_test = model.prepare_data(market_data, forward_periods=5, threshold=0.01)
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION MODEL PERFORMANCE")
    print("=" * 70)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Buy Signal Accuracy: {metrics.get('BUY_accuracy', 0):.4f} ({metrics.get('BUY_count', 0)} samples)")
    print(f"Hold Signal Accuracy: {metrics.get('HOLD_accuracy', 0):.4f} ({metrics.get('HOLD_count', 0)} samples)")
    print(f"Sell Signal Accuracy: {metrics.get('SELL_accuracy', 0):.4f} ({metrics.get('SELL_count', 0)} samples)")
    print("=" * 70)

    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))


def example_random_forest():
    """Example: Random Forest model."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2021-01-01", end="2023-12-31")

    logger.info("Creating and training random forest model...")
    model = MLTradingModel(model_type="rf")

    X_train, X_test, y_train, y_test = model.prepare_data(market_data, forward_periods=5, threshold=0.01)
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print("RANDOM FOREST MODEL PERFORMANCE")
    print("=" * 70)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Buy Signal Accuracy: {metrics.get('BUY_accuracy', 0):.4f} ({metrics.get('BUY_count', 0)} samples)")
    print(f"Hold Signal Accuracy: {metrics.get('HOLD_accuracy', 0):.4f} ({metrics.get('HOLD_count', 0)} samples)")
    print(f"Sell Signal Accuracy: {metrics.get('SELL_accuracy', 0):.4f} ({metrics.get('SELL_count', 0)} samples)")
    print("=" * 70)

    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))


def example_comparison():
    """Example: Compare models and backtest predictions."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2022-01-01", end="2023-12-31")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for model_type in ["logistic", "rf"]:
        logger.info(f"Training {model_type} model...")
        model = MLTradingModel(model_type=model_type)

        X_train, X_test, y_train, y_test = model.prepare_data(market_data, forward_periods=5, threshold=0.01)
        model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)

        print(f"\n{model_type.upper()} Accuracy: {metrics['accuracy']:.4f}")

        # Get predictions
        y_pred = model.predict(X_test)
        buy_signals = (y_pred == 1).sum()
        hold_signals = (y_pred == 0).sum()
        sell_signals = (y_pred == -1).sum()

        print(f"  Buy Signals: {buy_signals}")
        print(f"  Hold Signals: {hold_signals}")
        print(f"  Sell Signals: {sell_signals}")

    print("=" * 70)


def example_predict_next():
    """Example: Use trained model to predict next signal."""
    logger.info("Loading market data...")
    market_data = MarketDataLoader.load(symbol="SPY", start="2022-01-01", end="2023-12-31")

    logger.info("Training model...")
    model = MLTradingModel(model_type="logistic")

    X_train, X_test, y_train, y_test = model.prepare_data(market_data, forward_periods=5, threshold=0.01)
    model.train(X_train, y_train)

    print("\n" + "=" * 70)
    print("RECENT PREDICTIONS")
    print("=" * 70)

    # Get last few predictions
    recent_preds = model.predict(X_test[-10:])
    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    for i, pred in enumerate(recent_preds):
        print(f"Sample {i}: {signal_map[pred]}")

    print("=" * 70)


if __name__ == "__main__":
    print("Machine Learning Trading Models\n")

    try:
        example_logistic_regression()
    except Exception as e:
        logger.error(f"Logistic regression example failed: {e}")

    try:
        example_random_forest()
    except Exception as e:
        logger.error(f"Random forest example failed: {e}")

    try:
        example_comparison()
    except Exception as e:
        logger.error(f"Comparison example failed: {e}")

    try:
        example_predict_next()
    except Exception as e:
        logger.error(f"Predict next example failed: {e}")
