# 📈 Indian Equity Market Analyzer Pro

Professional-grade technical analysis tool for Indian stock market with advanced pattern detection based on master trader strategies (Dan Zanger & Qullamaggie).

## 🎯 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nifty-pattern.streamlit.app/)

## ✨ Features

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

## 🚀 Quick Start

### Option 1: Run Locally
```bash
# Clone the repository
git clone https://github.com/rxsinx/nifty-pattern.git
cd nifty-pattern

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
