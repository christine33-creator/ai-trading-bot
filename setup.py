from setuptools import setup, find_packages

setup(
    name="ai-trading-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "yfinance>=0.2.0",
    ],
)
