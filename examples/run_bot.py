"""Example runner for the AI trading bot."""
import logging

from bot import MarketDataLoader, TradingAgent, TradingEnvironment, extract_features, create_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    symbol = "SPY"
    raw_data = MarketDataLoader.load(symbol=symbol, start="2020-01-01")
    features = extract_features(raw_data)
    labels = create_labels(features)
    features = features.iloc[:-1]

    split = int(len(features) * 0.8)
    X_train = features.iloc[:split]
    y_train = labels.iloc[:split]
    X_test = features.iloc[split:]
    y_test = labels.iloc[split:]

    agent = TradingAgent()
    agent.train(X_train, y_train)

    actions = agent.predict(X_test)
    test_prices = raw_data.loc[X_test.index, "Close"]
    environment = TradingEnvironment(initial_cash=10000.0)
    backtest_df = environment.backtest(test_prices, actions)
    metrics = environment.evaluate(backtest_df)

    print("Backtest metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("Sample actions")
    print(backtest_df[["price", "action", "equity"]].head(10))


if __name__ == "__main__":
    main()
