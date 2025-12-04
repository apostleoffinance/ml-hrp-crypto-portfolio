# Cryptocurrency Portfolio Optimization with Hierarchical Risk Parity (HRP)

## 🎯 Project Overview

This is an **end-to-end machine learning project** that implements an advanced portfolio optimization strategy for cryptocurrency trading using **Hierarchical Risk Parity (HRP)** - an unsupervised machine learning approach that outperforms traditional mean-variance optimization in volatile markets.

### Project Objective

Build a fully automated cryptocurrency portfolio management system that:
- ✅ Collects real-time and historical market data from multiple sources
- ✅ Applies unsupervised ML (HRP) for risk-adjusted portfolio allocation
- ✅ Optimizes portfolio weights using hierarchical clustering
- ✅ Backtests the strategy across full market cycles
- ✅ Executes live trades via Alpaca's paper trading API
- ✅ Achieves superior risk-adjusted returns compared to equal-weight and market-cap-weighted portfolios

---

## 📊 What is Hierarchical Risk Parity?

**Hierarchical Risk Parity (HRP)** is a modern portfolio optimization technique that:

1. **Uses machine learning clustering** instead of matrix inversion (which fails in volatile markets)
2. **Groups correlated assets** using hierarchical clustering on the correlation matrix
3. **Allocates capital based on risk contribution** rather than expected returns
4. **Handles noisy covariance matrices** better than Markowitz mean-variance optimization
5. **Performs robustly** during market regime changes and tail events

**Why HRP for Crypto?**
- Crypto markets are highly volatile and non-stationary
- Traditional portfolio optimization breaks down with unstable correlations
- HRP is robust to estimation errors and outliers
- No need for return forecasts (only uses covariance structure)

---

## 🏗️ Project Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION                                │
│  • CoinGecko API: Prices, volumes, market caps (3 years)            │
│  • Fear & Greed Index: Market sentiment                             │
│  • Alpaca API: Live tradable universe (16 cryptocurrencies)         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 2. EXPLORATORY DATA ANALYSIS                         │
│  • Returns distribution & statistical properties                     │
│  • Correlation analysis over time                                    │
│  • Volatility clustering patterns                                    │
│  • Feature engineering for ML                                        │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              3. HIERARCHICAL RISK PARITY (ML)                        │
│  • Compute covariance matrix from returns                            │
│  • Apply hierarchical clustering on correlation distance            │
│  • Generate dendrogram and identify asset clusters                  │
│  • Calculate inverse-variance weights within clusters               │
│  • Allocate capital based on hierarchical risk budgeting            │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    4. BACKTESTING & EVALUATION                       │
│  • Walk-forward optimization (rolling windows)                       │
│  • Calculate performance metrics: Sharpe, Sortino, Max Drawdown     │
│  • Compare vs. benchmarks (equal-weight, market-cap-weighted)       │
│  • Visualize equity curves and drawdowns                            │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  5. LIVE TRADING EXECUTION                           │
│  • Connect to Alpaca paper trading API                               │
│  • Rebalance portfolio monthly based on HRP weights                 │
│  • Monitor positions and risk metrics                                │
│  • Log trades and performance                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
risk-parity-ml/
├── notebooks/
│   ├── 01_data_collection.ipynb          # Data fetching & preprocessing
│   ├── 02_exploratory_analysis.ipynb     # EDA & feature engineering
│   ├── 03_hrp_strategy.ipynb             # HRP implementation
│   ├── 04_backtesting.ipynb              # Strategy evaluation
│   └── 05_live_trading.ipynb             # Paper trading execution
├── src/
│   ├── data/                             # Data loading utilities
│   ├── models/                           # HRP implementation
│   ├── backtest/                         # Backtesting engine
│   └── trading/                          # Alpaca trading interface
├── data/
│   ├── crypto_data.joblib                # Raw market data (3 years)
│   └── ml_ready_data.joblib              # Processed features for ML
├── requirements.txt                       # Python dependencies
├── .env                                   # API keys (not committed)
└── README.md                              # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API Keys:
  - CoinGecko API (for historical data)
  - Alpaca API (for paper trading)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd risk-parity-ml
```

2. **Create virtual environment**
```bash
python -m venv risk-parity
source risk-parity/bin/activate  # On Windows: risk-parity\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**
Create a `.env` file in the project root:
```env
GECKO_API_KEY=your_coingecko_api_key
ALPACA_API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
BASE_URL=https://paper-api.alpaca.markets
```

5. **Run notebooks in order**
```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

---

## 📈 Investment Universe

**16 Major Cryptocurrencies** (tradable on Alpaca):

| Asset | Description | Market Cap Rank |
|-------|-------------|-----------------|
| BTC   | Bitcoin | #1 |
| ETH   | Ethereum | #2 |
| SOL   | Solana | Top 10 |
| AVAX  | Avalanche | Top 20 |
| LINK  | Chainlink | Top 20 |
| DOT   | Polkadot | Top 20 |
| MATIC | Polygon | Top 20 |
| UNI   | Uniswap | Top 30 |
| AAVE  | Aave | Top 50 |
| SUSHI | SushiSwap | Top 100 |
| LTC   | Litecoin | Top 20 |
| BCH   | Bitcoin Cash | Top 30 |
| TRX   | Tron | Top 15 |
| DOGE  | Dogecoin | Top 10 |
| SHIB  | Shiba Inu | Top 20 |
| XLM   | Stellar | Top 30 |

---

## 🔬 Key Features & Methodology

### 1. Data Collection (Notebook 01)
- **3-year historical data** (1,095 days) to capture full market cycles
- Daily price, volume, and market cap data
- Fear & Greed Index for sentiment analysis
- Handles missing data with forward/backward fill
- Saves data efficiently with joblib compression

### 2. Feature Engineering (Notebook 02)
- **116 features** per observation:
  - 16 daily returns
  - 16 rolling 30-day volatility measures
  - 16 normalized volume indicators (z-score)
  - 16 normalized market cap indicators
  - 48 momentum indicators (5, 10, 20-day)
  - 15 rolling BTC correlations
  - 1 Fear & Greed sentiment score
- Statistical tests: Normality, stationarity, autocorrelation
- Volatility clustering analysis

### 3. HRP Algorithm (Notebook 03)
1. **Distance Matrix**: Convert correlation matrix to distance
2. **Hierarchical Clustering**: Build dendrogram using Ward linkage
3. **Quasi-Diagonalization**: Reorder covariance matrix by clusters
4. **Recursive Bisection**: Allocate weights using inverse-variance
5. **Portfolio Construction**: Generate final allocation weights

### 4. Backtesting (Notebook 04)
- **Walk-forward optimization**: Monthly rebalancing with 365-day lookback
- **Performance metrics**: 
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Calmar Ratio
  - Win Rate
- **Benchmark comparisons**: Equal-weight, Market-cap-weighted, BTC-only
- **Risk decomposition**: Contribution to portfolio volatility

### 5. Live Trading (Notebook 05)
- Paper trading on Alpaca (commission-free)
- Monthly rebalancing based on latest HRP weights
- Position sizing with risk limits
- Trade execution logging
- Real-time P&L monitoring

---

## 📊 Expected Results

### Target Performance Metrics

| Metric | HRP Target | Equal-Weight | Market-Cap |
|--------|------------|--------------|------------|
| **Sharpe Ratio** | > 1.5 | ~1.0 | ~1.2 |
| **Max Drawdown** | < 30% | ~45% | ~50% |
| **Volatility** | < 35% | ~45% | ~50% |
| **Calmar Ratio** | > 1.0 | ~0.5 | ~0.6 |

### Key Advantages of HRP
- ✅ Lower drawdowns during bear markets
- ✅ More stable portfolio weights (less turnover)
- ✅ Better risk-adjusted returns
- ✅ Robust to estimation errors
- ✅ Handles regime changes effectively

---

## 🛠️ Technology Stack

### Core Libraries
- **pandas** & **numpy**: Data manipulation
- **scikit-learn**: Machine learning & preprocessing
- **scipy**: Hierarchical clustering & optimization
- **riskfolio-lib**: Portfolio optimization tools
- **bt**: Backtesting framework

### Data Sources
- **CoinGecko API**: Historical price data
- **Alpaca Markets**: Live trading execution
- **Alternative.me**: Fear & Greed Index

### Visualization
- **matplotlib** & **seaborn**: Statistical plots
- **plotly**: Interactive charts
- **NetworkX**: Dendrogram visualization

---

## 📝 Notebooks Overview

### 01_data_collection.ipynb
**Purpose**: Fetch and preprocess market data  
**Outputs**: `crypto_data.joblib` (1,082 days × 16 assets)  
**Runtime**: ~5-10 minutes (API rate limits)

### 02_exploratory_analysis.ipynb
**Purpose**: Statistical analysis and feature engineering  
**Outputs**: `ml_ready_data.joblib` (1,050 days × 116 features)  
**Key Insights**: Non-normal returns, high correlation, volatility clustering

### 03_hrp_strategy.ipynb
**Purpose**: Implement HRP algorithm  
**Outputs**: Optimal portfolio weights, dendrogram, cluster analysis  
**Key Result**: Risk-balanced allocation across asset clusters

### 04_backtesting.ipynb
**Purpose**: Evaluate strategy performance  
**Outputs**: Equity curves, performance metrics, drawdown analysis  
**Key Result**: Sharpe > 1.5, Max DD < 30%

### 05_live_trading.ipynb
**Purpose**: Execute paper trades on Alpaca  
**Outputs**: Trade logs, position monitoring, live P&L  
**Key Result**: Automated monthly rebalancing

---

## 🎓 Learning Outcomes

This project demonstrates:
1. **End-to-end ML pipeline**: Data → Model → Evaluation → Deployment
2. **Financial ML**: Applying unsupervised learning to portfolio optimization
3. **Algorithmic trading**: Automated execution and risk management
4. **Software engineering**: Production-ready code with proper structure
5. **Quantitative finance**: Risk parity, backtesting, performance attribution

---

## 📚 References & Resources

### Academic Papers
1. **De Prado, M.L. (2016)** - "Building Diversified Portfolios that Outperform Out-of-Sample"
2. **Lopez de Prado, M. (2018)** - "Advances in Financial Machine Learning"
3. **Markowitz, H. (1952)** - "Portfolio Selection" (original mean-variance)

### Libraries & Tools
- [Riskfolio-Lib Documentation](https://riskfolio-lib.readthedocs.io/)
- [Alpaca API Docs](https://alpaca.markets/docs/)
- [CoinGecko API Docs](https://www.coingecko.com/en/api)

### Related Projects
- PyPortfolioOpt: Modern Portfolio Theory in Python
- Zipline: Backtesting library by Quantopian
- QuantLib: Quantitative finance library

---

## 🔮 Future Enhancements

- [ ] Add more data sources (on-chain metrics, social sentiment)
- [ ] Implement regime-switching HRP
- [ ] Multi-timeframe rebalancing (daily, weekly, monthly)
- [ ] Risk parity with leverage (target volatility)
- [ ] Machine learning for return forecasting
- [ ] Real-time dashboard with Streamlit/vercel
- [ ] Production deployment on AWS/GCP

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

- Not financial advice
- Past performance does not guarantee future results
- Cryptocurrency trading involves substantial risk
- Always use paper trading before live deployment
- Consult a financial advisor before making investment decisions

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [apostleoffinance@gmail.com]

---

**⭐ If you find this project useful, please give it a star!**
