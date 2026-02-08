# 📈 Indian Equity Market Analyzer Pro

Professional-grade technical analysis tool for Indian stock market with advanced pattern detection based on master trader strategies (Dan Zanger & Qullamaggie).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

### 📊 Pattern Detection
- **11 Chart Patterns** with precise entry/exit points
- **Dan Zanger Patterns**: Cup & Handle, High Tight Flag, Ascending Triangle, Flat Base, Falling Wedge, Double Bottom
- **Qullamaggie Patterns**: Breakout, Episodic Pivot, Parabolic Short, Gap & Go, ABCD
- **Automatic Entry/Stop/Target Calculation**
- **Confidence Scoring** (HIGH/MEDIUM/LOW)

### 📈 Technical Analysis
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV
- **Multiple Timeframes**: 1M, 3M, 6M, 1Y, 2Y, 5Y, Max
- **Volume Profile Analysis**: POC, Value Area, High/Low Volume Nodes
- **Market Context**: Nifty 50 trend, India VIX, Sector Analysis

### 💰 Risk Management
- **Position Sizing**: 1% risk rule implementation
- **Multi-level Stop Loss**: Tight, Normal, Wide, Technical, Percentage
- **Profit Targets**: 3 levels with Risk/Reward ratios
- **Volatility-Adjusted Sizing**: ATR-based position adjustment
- **Max Drawdown Calculation**

### 📉 Advanced Charting
- **Interactive Candlestick Charts** with 200-bar history
- **Multiple Indicators Overlay**: SMA, EMA, Bollinger Bands, VWAP
- **MACD, RSI, Stochastic** subplots
- **Volume Profile** visualization
- **Support/Resistance Levels**

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/indian-equity-analyzer.git
cd indian-equity-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repository
5. Set main file: `app.py`
6. Click "Deploy"

## 📁 Project Structure

```
indian-equity-analyzer/
│
├── app.py                      # Main Streamlit application (900 lines)
├── pattern_detector.py         # Pattern detection module (900 lines)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── QUICKSTART.md              # 5-minute quick start guide
├── DEPLOYMENT.md              # Complete deployment guide
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
│
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## 💻 Usage

1. **Enter Stock Symbol**: Type NSE/BSE symbol (e.g., RELIANCE, TCS, HDFCBANK)
2. **Select Period**: Choose analysis timeframe (1mo to max)
3. **Set Portfolio Value**: For position sizing calculations
4. **Click Analyze**: Get comprehensive analysis

### Supported Stocks

- **All NSE Stocks**: Add `.NS` suffix (e.g., RELIANCE.NS)
- **All BSE Stocks**: Add `.BO` suffix (e.g., RELIANCE.BO)
- **Auto-detection**: Common stocks work without suffix

## 🎓 Trading Strategies

### Dan Zanger's Rules
- 8% Hard Stop Loss on all positions
- Volume Confirmation: 3x+ on breakouts
- Focus on liquid, high-volume leaders
- Patience in pattern formation

### Qullamaggie's Rules
- 1% Risk Rule: Never risk >1% per trade
- ORH Entry for momentum trades
- Quick profits, trail winners
- Trade market leaders only

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.31.0
- **Data**: yfinance (Yahoo Finance API)
- **Analysis**: pandas, numpy, scipy
- **Technical Indicators**: ta library
- **Visualization**: Plotly 5.18.0

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deploy to Streamlit Cloud
- **[PATTERN_DETECTOR_README.md](PATTERN_DETECTOR_README.md)** - Pattern module docs
- **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** - Complete setup guide
- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** - Architecture diagrams

## ⚠️ Disclaimer

**Educational purposes only. Not financial advice.**
- Trading involves risk of loss
- Past performance ≠ future results
- Always do your own research
- Consult a financial advisor

## 📄 License

MIT License - Free to use in personal and commercial projects

## 🙏 Credits

Pattern logic based on strategies from:
- **Dan Zanger**: Chart pattern master, world record holder
- **Kristjan Kullamägi (Qullamaggie)**: Swing trading expert
- **William O'Neil**: CANSLIM methodology

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


---

Last Updated: February 8, 2026
