import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
from functools import lru_cache
import io
from scipy import stats

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Equity Market Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 46px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .bullish {
        color: #00aa00;
        font-weight: bold;
        background-color: #e8f5e8;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .bearish {
        color: #ff4444;
        font-weight: bold;
        background-color: #ffe8e8;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .neutral {
        color: #ff9900;
        font-weight: bold;
        background-color: #fff4e8;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .pattern-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 18px !important;
        }
        .sub-header {
            font-size: 18px !important;
        }
        .stMetric {
            font-size: 18px !important;
        }
    }
    
    .stDataFrame {
        font-size: 18px !important;
    }
    
    .stPlotlyChart {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
    }
    
    .stock-ticker {
        font-size: 18px;
        font-weight: bold;
        color: #1a237e;
        padding: 10px;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class IndianEquityAnalyzer:
    """Master Trader Grade Analysis for Indian Equity Market - Enhanced Version"""
    
    def __init__(self, symbol, period='1y'):
        """Initialize with NSE/BSE symbol"""
        self.symbol = self.validate_stock_symbol(symbol)
        self.period = period
        self.data = None
        self.ticker = None
        
    def validate_stock_symbol(self, symbol):
        """Validate and format stock symbol"""
        valid_formats = ['.NS', '.BO']
        
        # Check if symbol has valid suffix
        if any(symbol.upper().endswith(suffix) for suffix in valid_formats):
            return symbol.upper()
        
        # Common Indian stocks for validation
        common_indian_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ICICIBANK', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
            'LT', 'AXISBANK', 'BAJFINANCE', 'WIPRO', 'HCLTECH',
            'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'ULTRACEMCO', 'TITAN',
            'ONGC', 'NTPC', 'POWERGRID', 'M&M', 'TATASTEEL'
        ]
        
        # Add .NS suffix for common Indian stocks
        if symbol.upper() in common_indian_stocks:
            return f"{symbol.upper()}.NS"
        
        return f"{symbol.upper()}.NS"  # Default to NSE
    
    @st.cache_data(ttl=3600, show_spinner="Fetching market data...")
    def fetch_data(_self, symbol, period):
        """Fetch and cache stock data"""
        analyzer = IndianEquityAnalyzer(symbol, period)
        success = analyzer._fetch_raw_data()
        return analyzer if success else None
    
    def _fetch_raw_data(self):
        """Fetch data from Yahoo Finance for Indian stocks"""
        try:
            self.ticker = yf.Ticker(self.symbol)
            self.data = self.ticker.history(period=self.period)
            
            if self.data.empty:
                # Try alternative suffix
                if self.symbol.endswith('.NS'):
                    alt_symbol = self.symbol.replace('.NS', '.BO')
                else:
                    alt_symbol = self.symbol.replace('.BO', '.NS')
                
                self.ticker = yf.Ticker(alt_symbol)
                self.data = self.ticker.history(period=self.period)
            
            if not self.data.empty:
                self.calculate_indicators()
                return True
            return False
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return False
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        df = self.data.copy()
        
        # Moving Averages
        df['SMA_9'] = ta.trend.sma_indicator(df['Close'], window=9)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_100'] = ta.trend.sma_indicator(df['Close'], window=100)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        df['EMA_8'] = ta.trend.ema_indicator(df['Close'], window=8)
        df['EMA_13'] = ta.trend.ema_indicator(df['Close'], window=13)
        df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
        df['EMA_34'] = ta.trend.ema_indicator(df['Close'], window=34)
        df['EMA_55'] = ta.trend.ema_indicator(df['Close'], window=55)
        df['EMA_70'] = ta.trend.ema_indicator(df['Close'], window=70)
        
        # MACD with multiple configurations
        macd_fast = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd_fast.macd()
        df['MACD_Signal'] = macd_fast.macd_signal()
        df['MACD_Hist'] = macd_fast.macd_diff()
        
        # RSI with multiple periods
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
        df['RSI_21'] = ta.momentum.rsi(df['Close'], window=21)
        
        # Bollinger Bands with multiple deviations
        bb_20_2 = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb_20_2.bollinger_hband()
        df['BB_Middle'] = bb_20_2.bollinger_mavg()
        df['BB_Lower'] = bb_20_2.bollinger_lband()
        
        bb_20_1 = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=1)
        df['BB_Upper_1'] = bb_20_1.bollinger_hband()
        df['BB_Lower_1'] = bb_20_1.bollinger_lband()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR and Volatility
        df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['ATR_7'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=7)
        
        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # VWAP (for intraday analysis)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Additional momentum indicators
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100  # Rate of Change
        df['Williams_%R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        
        # Price channels
        df['Donchian_High'] = df['High'].rolling(window=20).max()
        df['Donchian_Low'] = df['Low'].rolling(window=20).min()
        df['Donchian_Middle'] = (df['Donchian_High'] + df['Donchian_Low']) / 2
        
        self.data = df
    
    def detect_volume_profile(self):
        """Detect volume profile patterns with enhanced analysis"""
        df = self.data.tail(100)
        
        if len(df) < 20:
            return {}
        
        # Calculate price levels and volume distribution
        price_range = df['High'].max() - df['Low'].min()
        num_bins = 70  # More bins for finer analysis
        bins = np.linspace(df['Low'].min(), df['High'].max(), num_bins)
        
        volume_at_price = []
        value_area_volumes = []
        
        for i in range(len(bins) - 1):
            mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
            volume_sum = df[mask]['Volume'].sum()
            volume_at_price.append(volume_sum)
            
            # Calculate value traded (volume * average price)
            if mask.any():
                avg_price = df[mask]['Close'].mean()
                value_area_volumes.append(volume_sum * avg_price)
            else:
                value_area_volumes.append(0)
        
        volume_at_price = np.array(volume_at_price)
        value_area_volumes = np.array(value_area_volumes)
        
        # Find Point of Control (POC) - highest volume price level
        poc_idx = np.argmax(volume_at_price)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        poc_volume = volume_at_price[poc_idx]
        
        # Find Value Area (70% of volume)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * 0.70
        
        sorted_indices = np.argsort(volume_at_price)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_at_price[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        
        value_area_high = bins[max(value_area_indices) + 1]
        value_area_low = bins[min(value_area_indices)]
        
        # Calculate Value Area percentage
        value_area_percentage = (value_area_high - value_area_low) / price_range * 100
        
        # Identify high and low volume nodes
        volume_mean = volume_at_price.mean()
        volume_std = volume_at_price.std()
        
        high_volume_threshold = volume_mean + volume_std
        low_volume_threshold = volume_mean - volume_std
        
        high_volume_nodes = bins[:-1][volume_at_price > high_volume_threshold]
        low_volume_nodes = bins[:-1][volume_at_price < low_volume_threshold]
        
        # Calculate volume profile statistics
        try:
            volume_profile_stats = {
                'total_volume': total_volume,
                'volume_std': volume_std,
                'volume_skew': stats.skew(volume_at_price),
                'volume_kurtosis': stats.kurtosis(volume_at_price)
            }
        except:
            volume_profile_stats = {
                'total_volume': total_volume,
                'volume_std': volume_std,
                'volume_skew': 0,
                'volume_kurtosis': 0
            }
        
        # Identify single prints (low volume areas)
        single_prints = bins[:-1][volume_at_price < (volume_mean * 0.3)]
        
        return {
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'value_area_percentage': value_area_percentage,
            'high_volume_nodes': high_volume_nodes,
            'low_volume_nodes': low_volume_nodes,
            'single_prints': single_prints,
            'volume_distribution': volume_at_price,
            'value_distribution': value_area_volumes,
            'price_bins': bins,
            'stats': volume_profile_stats
        }
    
    def detect_chart_patterns(self):
      df = self.data.tail(100).copy()
      if len(df) < 50:
        return []
      detector = PatternDetector(df)
      return detector.detect_all_zanger_patterns()

    def detect_swing_patterns(self):
      df = self.data.tail(60).copy()
      if len(df) < 30:
        return []
      detector = PatternDetector(df)
      return detector.detect_all_swing_patterns()

  
    def get_trading_signal(self):
        """Generate comprehensive trading signal with weighted scoring"""
        df = self.data
        if df is None or len(df) < 50:
            return "NO DATA", [], 0, "gray"
        
        current = df.iloc[-1]
        
        signals = []
        score = 0
        
        # Weighted scoring system
        weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volume': 0.2,
            'volatility': 0.15,
            'pattern': 0.1
        }
        
        # 1. Trend Analysis (30%)
        trend_score = 0
        trend_signals = []
        
        if current['Close'] > current['SMA_20']:
            trend_signals.append("✅ Price above 20 SMA (Short-term bullish)")
            trend_score += 1
        else:
            trend_signals.append("❌ Price below 20 SMA (Short-term bearish)")
            trend_score -= 1
        
        if current['Close'] > current['SMA_50']:
            trend_signals.append("✅ Price above 50 SMA (Medium-term bullish)")
            trend_score += 2
        else:
            trend_signals.append("❌ Price below 50 SMA (Medium-term bearish)")
            trend_score -= 2
        
        if current['Close'] > current['SMA_200']:
            trend_signals.append("✅ Price above 200 SMA (Long-term bullish)")
            trend_score += 3
        else:
            trend_signals.append("❌ Price below 200 SMA (Long-term bearish)")
            trend_score -= 3
        
        # EMA alignment check
        if (current['EMA_8'] > current['EMA_21'] > current['EMA_55'] and
            current['Close'] > current['EMA_8']):
            trend_signals.append("✅ All EMAs aligned bullish (8 > 21 > 55)")
            trend_score += 2
        
        trend_normalized = trend_score / 8  # Normalize to -1 to 1
        score += trend_normalized * weights['trend'] * 100
        
        # 2. Momentum Analysis (25%)
        momentum_score = 0
        momentum_signals = []
        
        # MACD
        if current['MACD'] > current['MACD_Signal']:
            momentum_signals.append("✅ MACD bullish (Above signal line)")
            momentum_score += 2
        else:
            momentum_signals.append("❌ MACD bearish (Below signal line)")
            momentum_score -= 2
        
        # RSI
        if current['RSI_14'] > 70:
            momentum_signals.append("⚠️ RSI Overbought (>70)")
            momentum_score -= 1
        elif current['RSI_14'] < 30:
            momentum_signals.append("✅ RSI Oversold (<30)")
            momentum_score += 2
        else:
            momentum_signals.append(f"✅ RSI Neutral ({current['RSI_14']:.1f})")
        
        # Stochastic
        if current['Stoch_K'] > 80 and current['Stoch_D'] > 80:
            momentum_signals.append("⚠️ Stochastic overbought")
            momentum_score -= 1
        elif current['Stoch_K'] < 20 and current['Stoch_D'] < 20:
            momentum_signals.append("✅ Stochastic oversold")
            momentum_score += 1
        
        momentum_normalized = momentum_score / 5
        score += momentum_normalized * weights['momentum'] * 100
        
        # 3. Volume Analysis (20%)
        volume_score = 0
        volume_signals = []
        
        if current['Volume'] > current['Volume_SMA_20'] * 1.5:
            volume_signals.append("✅ High volume (Strong interest)")
            volume_score += 2
        elif current['Volume'] < current['Volume_SMA_20'] * 0.5:
            volume_signals.append("⚠️ Very low volume (Caution)")
            volume_score -= 1
        else:
            volume_signals.append("✅ Average volume")
        
        # OBV trend
        if len(df) > 20:
            obv_trend = np.polyfit(range(20), df['OBV'].tail(20).values, 1)[0]
            if obv_trend > 0:
                volume_signals.append("✅ OBV trending up (Accumulation)")
                volume_score += 1
            else:
                volume_signals.append("❌ OBV trending down (Distribution)")
                volume_score -= 1
        
        volume_normalized = volume_score / 3
        score += volume_normalized * weights['volume'] * 100
        
        # 4. Volatility Analysis (15%)
        volatility_score = 0
        volatility_signals = []
        
        # ATR relative position
        atr_percent = current['ATR_14'] / current['Close'] * 100
        if atr_percent > 3:
            volatility_signals.append(f"⚠️ High volatility (ATR: {atr_percent:.1f}%)")
            volatility_score -= 1
        elif atr_percent < 1:
            volatility_signals.append(f"✅ Low volatility (ATR: {atr_percent:.1f}%)")
            volatility_score += 1
        else:
            volatility_signals.append(f"✅ Normal volatility (ATR: {atr_percent:.1f}%)")
        
        # Bollinger Band position
        bb_position = (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
        if bb_position > 0.8:
            volatility_signals.append("⚠️ Near BB upper band (Overbought)")
            volatility_score -= 1
        elif bb_position < 0.2:
            volatility_signals.append("✅ Near BB lower band (Oversold)")
            volatility_score += 1
        
        volatility_normalized = volatility_score / 2
        score += volatility_normalized * weights['volatility'] * 100
        
        # 5. Pattern Detection (10%)
        pattern_score = 0
        pattern_signals = []
        
        patterns = self.detect_chart_patterns() + self.detect_swing_patterns()
        bullish_patterns = sum(1 for p in patterns if p['signal'] == 'BULLISH')
        bearish_patterns = sum(1 for p in patterns if p['signal'] == 'BEARISH')
        
        if bullish_patterns > bearish_patterns:
            pattern_signals.append(f"✅ {bullish_patterns} bullish patterns detected")
            pattern_score += 2
        elif bearish_patterns > bullish_patterns:
            pattern_signals.append(f"❌ {bearish_patterns} bearish patterns detected")
            pattern_score -= 2
        
        pattern_normalized = pattern_score / 2
        score += pattern_normalized * weights['pattern'] * 100
        
        # Combine all signals
        signals.extend(trend_signals)
        signals.extend(momentum_signals)
        signals.extend(volume_signals)
        signals.extend(volatility_signals)
        signals.extend(pattern_signals)
        
        # Determine overall signal
        if score >= 70:
            overall = "🟢 STRONG BUY"
            color = "green"
        elif score >= 50:
            overall = "🟢 BUY"
            color = "lightgreen"
        elif score >= 30:
            overall = "🟡 ACCUMULATE"
            color = "orange"
        elif score >= 10:
            overall = "🟡 HOLD"
            color = "yellow"
        elif score >= -10:
            overall = "🟡 NEUTRAL"
            color = "gray"
        elif score >= -30:
            overall = "🔴 SELL"
            color = "pink"
        else:
            overall = "🔴 STRONG SELL"
            color = "red"
        
        return overall, signals, score, color
    
    def get_risk_management(self, portfolio_value=1000000):
        """Calculate advanced risk management parameters"""
        df = self.data
        if df is None or len(df) < 20:
            return {}
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR_14'].iloc[-1]
        
        # Multi-level stop loss strategy
        stop_loss_levels = {
            'tight': current_price - (1 * atr),  # 1 ATR stop (aggressive)
            'normal': current_price - (1.5 * atr),  # 1.5 ATR stop (balanced)
            'wide': current_price - (2 * atr),  # 2 ATR stop (conservative)
            'percentage': current_price * 0.97,  # 3% hard stop
            'technical': min(
                df['Low'].tail(20).min(),
                df['BB_Lower'].iloc[-1],
                df['SMA_20'].iloc[-1] * 0.98
            )
        }
        
        recommended_stop = max(
            stop_loss_levels['technical'],
            stop_loss_levels['percentage'],
            stop_loss_levels['normal']
        )
        
        # Multi-level profit targets
        profit_targets = {
            'target_1': current_price * 1.10,  # 10% (quick profit)
            'target_2': current_price * 1.20,  # 20% (primary target)
            'target_3': current_price * 1.35,  # 35% (runner)
            'fib_161': current_price * 1.618,  # Fibonacci extension
            'measured_move': current_price + (current_price - recommended_stop) * 2  # 2:1 measured move
        }
        
        # Position sizing based on risk
        risk_per_share = current_price - recommended_stop
        
        # Calculate position size (1% portfolio risk rule)
        max_risk_amount = portfolio_value * 0.01
        
        if risk_per_share > 0:
            position_size = int(max_risk_amount / risk_per_share)
            position_value = position_size * current_price
            
            # Additional constraints
            max_position_value = portfolio_value * 0.25  # Max 25% in one position
            if position_value > max_position_value:
                position_size = int(max_position_value / current_price)
                position_value = position_size * current_price
            
            actual_risk = position_size * risk_per_share
            portfolio_risk_percent = (actual_risk / portfolio_value) * 100
        else:
            position_size = 0
            position_value = 0
            actual_risk = 0
            portfolio_risk_percent = 0
        
        # Risk/Reward ratios
        rr_ratios = {}
        for target_name, target_price in profit_targets.items():
            if risk_per_share > 0:
                reward = target_price - current_price
                rr_ratios[target_name] = reward / risk_per_share
            else:
                rr_ratios[target_name] = 0
        
        # Volatility-adjusted position sizing
        volatility_score = atr / current_price * 100
        position_adjustment = 1.0
        
        if volatility_score > 4:
            position_adjustment = 0.7  # Reduce position by 30% for high volatility
        elif volatility_score < 1.5:
            position_adjustment = 1.2  # Increase position by 20% for low volatility
        
        adjusted_position_size = int(position_size * position_adjustment)
        
        return {
            'entry_price': current_price,
            'stop_loss': recommended_stop,
            'stop_loss_levels': stop_loss_levels,
            'profit_targets': profit_targets,
            'risk_per_share': risk_per_share,
            'position_size': adjusted_position_size,
            'position_value': adjusted_position_size * current_price,
            'portfolio_risk_percent': portfolio_risk_percent,
            'risk_reward_ratios': rr_ratios,
            'atr_percent': volatility_score,
            'position_adjustment': position_adjustment,
            'max_drawdown': (current_price - recommended_stop) / current_price * 100
        }
    
    def get_market_context(self):
        """Analyze broader market conditions"""
        try:
            # Nifty 50
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period='3mo')
            
            # India VIX
            vix = yf.Ticker("^INDIAVIX")
            vix_data = vix.history(period='1mo')
            
            if nifty_data.empty:
                return {}
            
            current_nifty = nifty_data['Close'].iloc[-1]
            nifty_sma_50 = nifty_data['Close'].rolling(window=50).mean().iloc[-1]
            nifty_sma_200 = nifty_data['Close'].rolling(window=200).mean().iloc[-1]
            
            nifty_trend = "BULLISH" if current_nifty > nifty_sma_50 and current_nifty > nifty_sma_200 else "BEARISH"
            
            current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else None
            
            return {
                'nifty_level': current_nifty,
                'nifty_change': ((current_nifty - nifty_data['Close'].iloc[0]) / nifty_data['Close'].iloc[0]) * 100,
                'nifty_trend': nifty_trend,
                'above_50_sma': current_nifty > nifty_sma_50,
                'above_200_sma': current_nifty > nifty_sma_200,
                'vix': current_vix,
                'market_condition': 'High Volatility' if current_vix and current_vix > 20 else 'Normal',
                'vix_trend': 'FEAR' if current_vix and current_vix > 25 else 'GREED' if current_vix and current_vix < 15 else 'NEUTRAL'
            }
        except:
            return {}
    
    def get_sector_analysis(self):
        """Get sector and peer analysis"""
        try:
            info = self.ticker.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            
            # Sector ETF mapping for Indian markets
            sector_etfs = {
                'Technology': 'INFY.NS',
                'Financial Services': 'HDFCBANK.NS',
                'Energy': 'RELIANCE.NS',
                'Healthcare': 'SUNPHARMA.NS',
                'Consumer Defensive': 'ITC.NS',
                'Industrials': 'LT.NS',
                'Communication Services': 'BHARTIARTL.NS',
                'Basic Materials': 'TATASTEEL.NS',
                'Utilities': 'NTPC.NS',
                'Real Estate': 'DLF.NS',
                'Consumer Cyclical': 'MARUTI.NS'
            }
            
            if sector in sector_etfs:
                sector_ticker = yf.Ticker(sector_etfs[sector])
                sector_data = sector_ticker.history(period=self.period)
                
                if not sector_data.empty and not self.data.empty:
                    stock_return = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100
                    sector_return = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[0] - 1) * 100
                    
                    # Calculate relative strength
                    relative_strength = stock_return - sector_return
                    
                    # Beta calculation (simplified)
                    if len(self.data) > 20 and len(sector_data) > 20:
                        stock_returns = self.data['Close'].pct_change().dropna()
                        sector_returns = sector_data['Close'].pct_change().dropna()
                        
                        # Align data
                        common_index = stock_returns.index.intersection(sector_returns.index)
                        if len(common_index) > 10:
                            stock_returns_aligned = stock_returns.loc[common_index]
                            sector_returns_aligned = sector_returns.loc[common_index]
                            
                            # Calculate beta
                            covariance = np.cov(stock_returns_aligned, sector_returns_aligned)[0, 1]
                            variance = np.var(sector_returns_aligned)
                            beta = covariance / variance if variance != 0 else 1.0
                        else:
                            beta = 1.0
                    else:
                        beta = 1.0
                    
                    return {
                        'sector': sector,
                        'industry': industry,
                        'stock_return': stock_return,
                        'sector_return': sector_return,
                        'outperformance': relative_strength,
                        'relative_strength': 'OUTPERFORMING' if relative_strength > 5 else 'UNDERPERFORMING' if relative_strength < -5 else 'IN-LINE',
                        'beta': beta,
                        'risk_category': 'High Beta' if beta > 1.3 else 'Low Beta' if beta < 0.7 else 'Market Beta'
                    }
        except:
            pass
        
        return None
    
    def get_company_info(self):
        """Get comprehensive company fundamental information"""
        info = {}
        try:
            ticker_info = self.ticker.info
            
            # Basic info
            info['name'] = ticker_info.get('longName', self.symbol)
            info['sector'] = ticker_info.get('sector', 'N/A')
            info['industry'] = ticker_info.get('industry', 'N/A')
            
            # Financial metrics
            market_cap = ticker_info.get('marketCap', 0)
            info['market_cap'] = market_cap
            info['market_cap_formatted'] = self.format_market_cap(market_cap)
            
            info['pe_ratio'] = ticker_info.get('trailingPE', 'N/A')
            info['forward_pe'] = ticker_info.get('forwardPE', 'N/A')
            info['pb_ratio'] = ticker_info.get('priceToBook', 'N/A')
            info['peg_ratio'] = ticker_info.get('pegRatio', 'N/A')
            
            # Profitability
            info['roe'] = ticker_info.get('returnOnEquity', 'N/A')
            info['roa'] = ticker_info.get('returnOnAssets', 'N/A')
            info['profit_margin'] = ticker_info.get('profitMargins', 'N/A')
            
            # Growth
            info['revenue_growth'] = ticker_info.get('revenueGrowth', 'N/A')
            info['earnings_growth'] = ticker_info.get('earningsGrowth', 'N/A')
            
            # Dividends
            div_yield = ticker_info.get('dividendYield', 0)
            info['dividend_yield'] = div_yield * 100 if div_yield else 0
            info['payout_ratio'] = ticker_info.get('payoutRatio', 'N/A')
            
            # Valuation
            info['eps'] = ticker_info.get('trailingEps', 'N/A')
            info['forward_eps'] = ticker_info.get('forwardEps', 'N/A')
            info['book_value'] = ticker_info.get('bookValue', 'N/A')
            
            # Price information
            info['52w_high'] = ticker_info.get('fiftyTwoWeekHigh', 0)
            info['52w_low'] = ticker_info.get('fiftyTwoWeekLow', 0)
            info['beta'] = ticker_info.get('beta', 1.0)
            
            # Additional metrics
            info['debt_to_equity'] = ticker_info.get('debtToEquity', 'N/A')
            info['current_ratio'] = ticker_info.get('currentRatio', 'N/A')
            info['quick_ratio'] = ticker_info.get('quickRatio', 'N/A')
            
            # Institutional info
            info['held_by_institutions'] = ticker_info.get('heldPercentInstitutions', 'N/A')
            info['short_ratio'] = ticker_info.get('shortRatio', 'N/A')
            
            # Description
            info['description'] = ticker_info.get('longBusinessSummary', 'No description available.')
            
        except Exception as e:
            st.warning(f"Could not fetch complete company info: {str(e)}")
        
        return info

    def get_analyst_forecasts(self):
        """
        Get comprehensive analyst recommendations, price targets, and estimates
        Enhanced version with earnings forecasts, revenue estimates, and recommendations trend
        """
        try:
            info = self.ticker.info
            current_price = self.data['Close'].iloc[-1] if self.data is not None and not self.data.empty else 0
            
            # Initialize comprehensive analyst data structure
            analyst_data = {
                'current_price': current_price,
                'target_mean': info.get('targetMeanPrice'),
                'target_high': info.get('targetHighPrice'),
                'target_low': info.get('targetLowPrice'),
                'target_median': info.get('targetMedianPrice'),
                'recommendation': info.get('recommendationKey', 'hold'),
                'num_analysts': info.get('numberOfAnalystOpinions', 0),
                'current_year_eps': info.get('epsCurrentYear'),
                'next_year_eps': info.get('epsForward'),
                'eps_growth': info.get('earningsGrowth'),
                'revenue_growth': info.get('revenueGrowth'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'upside_percent': None,
                'risk_rating': 'N/A',
                'recommendation_text': 'N/A',
                'target_range': None,
                'target_range_percent': None,
                'recent_recommendations': [],
                'recommendation_trend': {},
                'consensus_score': None
            }
        
            # Calculate upside
            if analyst_data['target_mean'] and current_price > 0:
                upside = ((analyst_data['target_mean'] - current_price) / current_price) * 100
                analyst_data['upside_percent'] = round(upside, 2)
                
                if upside > 20:
                    analyst_data['risk_rating'] = 'High Reward'
                elif upside > 10:
                    analyst_data['risk_rating'] = 'Moderate Reward'
                elif upside > 0:
                    analyst_data['risk_rating'] = 'Low Reward'
                else:
                    analyst_data['risk_rating'] = 'Overvalued'

            # Format recommendation
            rec_map = {
                'strong_buy': 'Strong Buy', 'buy': 'Buy', 'outperform': 'Outperform',
                'hold': 'Hold', 'underperform': 'Underperform', 'sell': 'Sell'
            }
            rec_key = analyst_data['recommendation']
            if rec_key:
                analyst_data['recommendation_text'] = rec_map.get(rec_key.lower(), rec_key.title())
            else:
                analyst_data['recommendation_text'] = 'N/A'
            
            # Calculate target range
            if analyst_data['target_high'] and analyst_data['target_low']:
                target_range = analyst_data['target_high'] - analyst_data['target_low']
                analyst_data['target_range'] = target_range
                if analyst_data['target_mean']:
                    analyst_data['target_range_percent'] = (target_range / analyst_data['target_mean']) * 100
            
            # Try to get recommendations trend
            try:
                recommendations = self.ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    recent = recommendations.tail(10)
                    analyst_data['recent_recommendations'] = recent.to_dict('records')
                    
                    if 'To Grade' in recent.columns:
                        rec_counts = recent['To Grade'].value_counts().to_dict()
                        analyst_data['recommendation_trend'] = rec_counts
                        
                        grade_weights = {
                            'Strong Buy': 5, 'Buy': 4, 'Outperform': 4,
                            'Hold': 3, 'Neutral': 3, 'Underperform': 2, 'Sell': 1
                        }
                        total_score = sum(grade_weights.get(grade, 3) * count for grade, count in rec_counts.items())
                        total_recs = sum(rec_counts.values())
                        if total_recs > 0:
                            analyst_data['consensus_score'] = round(total_score / total_recs, 2)
            except:
                pass
        
            # Try to get earnings calendar
            try:
                calendar = self.ticker.calendar
                if calendar is not None and not calendar.empty:
                    if len(calendar) > 0:
                        if 'Earnings Date' in calendar.columns:
                            analyst_data['next_earnings_date'] = str(calendar['Earnings Date'].iloc[0])[:10]
                        if 'Earnings Average' in calendar.columns:
                            analyst_data['earnings_estimate'] = calendar['Earnings Average'].iloc[0]
            except:
                pass
            
            return analyst_data
        
        except Exception as e:
            # Fallback to basic data
            try:
                info = self.ticker.info
                current_price = self.data['Close'].iloc[-1] if self.data is not None and not self.data.empty else 0
                
                return {
                    'current_price': current_price,
                    'target_mean': info.get('targetMeanPrice'),
                    'target_high': info.get('targetHighPrice'),
                    'target_low': info.get('targetLowPrice'),
                    'recommendation': info.get('recommendationKey', 'N/A'),
                    'recommendation_text': info.get('recommendationKey', 'N/A').title() if info.get('recommendationKey') else 'N/A',
                    'num_analysts': info.get('numberOfAnalystOpinions', 0),
                    'upside_percent': None,
                    'error': str(e)
                }
            except:
                return {
                    'current_price': self.data['Close'].iloc[-1] if self.data is not None and not self.data.empty else 0,
                    'target_mean': None,
                    'recommendation_text': 'Not Available',
                    'num_analysts': 0,
                    'upside_percent': None
                }
    
    def format_market_cap(self, market_cap):
        """Format market cap to readable string"""
        if market_cap >= 1e12:
            return f"₹{market_cap/1e12:.2f} Lac Cr"
        elif market_cap >= 1e10:
            return f"₹{market_cap/1e10:.2f} K Cr"
        elif market_cap >= 1e7:
            return f"₹{market_cap/1e7:.2f} Cr"
        else:
            return f"₹{market_cap:,.0f}"

def create_candlestick_chart(analyzer):
    """Create advanced candlestick chart with multiple indicators"""
    df = analyzer.data.tail(200)
    
    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.2],
        subplot_titles=('Price Action with Indicators', 'MACD', 'RSI', 'Stochastic', 'Volume Profile')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#17e817',
            decreasing_line_color='#f55433'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                            line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                            line=dict(color='blue', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', 
                            line=dict(color='red', width=2)), row=1, col=1)
    
    # Exponential Moving Averages
    
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', 
                            line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_70'], name='EMA 70', 
                            line=dict(color='brown', width=1, dash='dash')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                            line=dict(color='gray', width=1, dash='dot'), fillcolor='rgba(128,128,128,0.2)',
                            fill='tonexty'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                            line=dict(color='gray', width=1, dash='dot'), fill='tonexty'), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', 
                            line=dict(color='magenta', width=1.5, dash='dashdot')), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                            line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                            line=dict(color='red', width=1.5)), row=2, col=1)
    
    # MACD Histogram
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', 
                        marker_color=colors, opacity=0.6), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14', 
                            line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="gray", opacity=0.1, row=3, col=1)
    
    # Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K', 
                            line=dict(color='blue', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D', 
                            line=dict(color='red', width=1.5)), row=4, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    # Volume
    colors_vol = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350' 
                  for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                        marker_color=colors_vol, opacity=0.7), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_SMA_20'], name='Vol SMA 20', 
                            line=dict(color='orange', width=1.5)), row=5, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'',
        xaxis_rangeslider_visible=False,
        height=1400,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Date", row=5, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", row=4, col=1)
    fig.update_yaxes(title_text="Volume", row=5, col=1)
    
    return fig

def create_volume_profile_chart(analyzer):
    """Create comprehensive volume profile chart"""
    vp = analyzer.detect_volume_profile()
    
    if not vp:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'domain'}]],
        subplot_titles=('Volume Profile', 'Value Area Distribution'),
        column_widths=[0.7, 0.3]
    )
    
    # Volume Profile Bars
    bin_centers = (vp['price_bins'][:-1] + vp['price_bins'][1:]) / 2
    
    fig.add_trace(go.Bar(
        y=bin_centers,
        x=vp['volume_distribution'],
        orientation='h',
        name='Volume',
        marker=dict(
            color=vp['volume_distribution'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Volume")
        )
    ), row=1, col=1)
    
    # Point of Control
    fig.add_hline(y=vp['poc_price'], line_dash="solid", line_color="red", 
                  line_width=3, row=1, col=1,
                  annotation_text=f"POC: ₹{vp['poc_price']:.2f}")
    
    # Value Area
    fig.add_hrect(y0=vp['value_area_low'], y1=vp['value_area_high'], 
                  line_width=0, fillcolor="red", opacity=0.2,
                  annotation_text=f"Value Area ({vp['value_area_percentage']:.1f}%)",
                  annotation_position="right", row=1, col=1)
    
    # Current Price
    current_price = analyzer.data['Close'].iloc[-1]
    fig.add_hline(y=current_price, line_dash="dash", line_color="green",
                  line_width=2, row=1, col=1,
                  annotation_text=f"Current: ₹{current_price:.2f}")
    
    # Value Distribution (Pie Chart)
    value_labels = ['High Volume Nodes', 'Value Area', 'Low Volume Nodes']
    
    # Calculate percentages
    high_volume_value = np.sum(vp['value_distribution'][vp['volume_distribution'] > 
                                                       np.percentile(vp['volume_distribution'], 70)])
    low_volume_value = np.sum(vp['value_distribution'][vp['volume_distribution'] < 
                                                      np.percentile(vp['volume_distribution'], 30)])
    other_value = np.sum(vp['value_distribution']) - high_volume_value - low_volume_value
    
    values = [high_volume_value, other_value, low_volume_value]

    fig.add_trace(go.Pie(
        labels=value_labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#00cc96', '#636efa', '#ef553b']),
        textinfo='percent+label'
     ), row=1, col=2)
    
    fig.update_layout(
        title='Volume Profile Analysis',
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Volume', row=1, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    
    return fig

def create_pattern_summary_card(pattern):
    """Create a styled card for pattern display"""
    if pattern['signal'] == 'BULLISH':
        color_class = "bullish"
        icon = "📈"
    elif pattern['signal'] == 'BEARISH':
        color_class = "bearish"
        icon = "📉"
    else:
        color_class = "neutral"
        icon = "⚖️"
    
    card_html = f"""
    <div class="pattern-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: white;">{icon} {pattern['pattern']}</h3>
                <p style="margin: 5px 0; color: #f0f0f0;">Signal: <span class="{color_class}">{pattern['signal']}</span></p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; color: #f0f0f0;">Confidence: {pattern['confidence']}</p>
                <p style="margin: 0; color: #f0f0f0;">Score: {pattern['score']:.2f}</p>
            </div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
        <p style="margin: 5px 0; color: #f0f0f0;"><strong>Description:</strong> {pattern['description']}</p>
        <p style="margin: 5px 0; color: #f0f0f0;"><strong>Action:</strong> {pattern['action']}</p>
    </div>
    """
    return card_html

def export_analysis_report(analyzer, patterns, risk_params, signal_info):
    """Export analysis to CSV format"""
    import io
    
    # Create summary DataFrame
    summary_data = {
        'Symbol': [analyzer.symbol],
        'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Current Price': [analyzer.data['Close'].iloc[-1] if analyzer.data is not None else 'N/A'],
        'Trading Signal': [signal_info[0]],
        'Signal Score': [signal_info[2]],
        'Patterns Detected': [', '.join([p['pattern'] for p in patterns]) if patterns else 'None'],
        'Stop Loss': [risk_params.get('stop_loss', 'N/A')],
        'Position Size': [risk_params.get('position_size', 'N/A')],
        'Risk/Reward 1:': [risk_params.get('risk_reward_ratios', {}).get('target_1', 'N/A')],
        'Portfolio Risk %': [risk_params.get('portfolio_risk_percent', 'N/A')],
        'ATR %': [risk_params.get('atr_percent', 'N/A')],
        'Max Drawdown %': [risk_params.get('max_drawdown', 'N/A')]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return csv_content

def main():
    """Main Streamlit application"""
    
    st.markdown('<div class="main-header">🎯 Indian Equity Market Analyzer Pro</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Master Trader Grade Analysis - Dan Zanger & Qullamaggie Strategies</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="stock-ticker">📊</div>', unsafe_allow_html=True)
        
        st.header("Analysis Settings")
        
        # Stock selection
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="",
            help="Enter NSE symbol (e.g., RELIANCE, TCS, HDFCBANK)"
        )
        
        # Analysis period
        period = st.selectbox(
            "Analysis Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            index=3
        )
        
        # Portfolio size for risk management
        portfolio_value = st.number_input(
            "Portfolio Value (₹)",
            min_value=10000,
            max_value=10000000,
            value=1000000,
            step=10000,
            help="Used for position sizing calculations"
        )
        
        # Analysis options
        st.markdown("---")
        st.markdown("### Analysis Options")
        
        show_advanced = st.checkbox("Show Advanced Analysis", value=True)
        show_patterns = st.checkbox("Show Pattern Detection", value=True)
        show_risk = st.checkbox("Show Risk Management", value=True)
        show_market = st.checkbox("Show Market Context", value=True)
        
        # Analyze button
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🔄 Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Pattern Library")
        
        with st.expander("Dan Zanger Patterns"):
            st.markdown("""
            - **Cup and Handle**: Most reliable bullish pattern
            - **High Tight Flag**: Explosive breakout pattern
            - **Ascending Triangle**: Bullish continuation
            - **Flat Base**: Consolidation before breakout
            - **Falling Wedge**: Bullish reversal
            """)
        
        with st.expander("Qullamaggie Patterns"):
            st.markdown("""
            - **Breakout**: ORH entry with volume
            - **Episodic Pivot**: Gap and go momentum
            - **Parabolic Short**: Reversion to mean setup
            - **Gap and Go**: Continuation pattern
            - **ABCD Pattern**: Harmonic trading
            """)
        
        st.markdown("---")
        st.markdown("### 🎓 Trading Rules")
        
        st.info("""
        **Zanger's Golden Rules:**
        1. Volume confirms every breakout
        2. 8% absolute stop loss
        3. Focus on liquid leaders
        4. Patience in pattern formation
        
        **Qullamaggie's Discipline:**
        1. Never risk >1% per trade
        2. Trade market leaders only
        3. ORH entry for momentum
        4. Quick profits, trail winners
        """)
    
    if analyze_btn and symbol:
        with st.spinner(f'🔄 Analyzing {symbol}...'):
            # Create analyzer instance and fetch data
            analyzer = IndianEquityAnalyzer(symbol, period)
            
            if analyzer._fetch_raw_data():
                # Company Information Section
                st.markdown('<div class="sub-header">🏢 Company Overview</div>', unsafe_allow_html=True)
                
                info = analyzer.get_company_info()
                company_info = st.container()
                
                with company_info:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Company", info.get('name', symbol))
                        st.metric("Sector", info.get('sector', 'N/A'))
                        
                    with col2:
                        st.metric("Market Cap", info.get('market_cap_formatted', 'N/A'))
                        st.metric("Industry", info.get('industry', 'N/A'))
                        
                    with col3:
                        pe = info.get('pe_ratio', 'N/A')
                        st.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else 'N/A')
                        st.metric("P/B Ratio", f"{info.get('pb_ratio', 'N/A'):.2f}" 
                                 if isinstance(info.get('pb_ratio'), (int, float)) else 'N/A')
                        
                    with col4:
                        st.metric("Beta", f"{info.get('beta', 1.0):.2f}")
                        div_yield = info.get('dividend_yield', 0)
                        st.metric("Div Yield", f"{div_yield:.2f}%" if div_yield else '0.00%')
                
                # Current Price Information
                current = analyzer.data.iloc[-1]
                prev = analyzer.data.iloc[-2]
                change = current['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100
                
                st.markdown('<div class="sub-header">📊 Current Market Data</div>', unsafe_allow_html=True)
                
                price_cols = st.columns(5)
                with price_cols[0]:
                    st.metric(
                        "Current Price",
                        f"₹{current['Close']:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
                with price_cols[1]:
                    st.metric("Day High", f"₹{current['High']:.2f}")
                with price_cols[2]:
                    st.metric("Day Low", f"₹{current['Low']:.2f}")
                with price_cols[3]:
                    st.metric("52W High", f"₹{info.get('52w_high', 0):.2f}")
                with price_cols[4]:
                    st.metric("52W Low", f"₹{info.get('52w_low', 0):.2f}")

                # ==================== ADVANCED ANALYST FORECASTS SECTION ====================
                
                st.markdown('<div class="sub-header">📊 Analyst Forecasts & Estimates</div>', unsafe_allow_html=True)
                
                try:
                    forecasts = analyzer.get_analyst_forecasts()
                    
                    # Check if we have meaningful data
                    if forecasts.get('num_analysts', 0) > 0 or forecasts.get('target_mean'):
                        
                        # ========== ROW 1: Price Targets & Recommendations ==========
                        st.markdown("#### 🎯 Price Targets & Analyst Consensus")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            rec_text = forecasts.get('recommendation_text', 'N/A')
                            # Color code the recommendation
                            if rec_text in ['Strong Buy', 'Buy', 'Outperform']:
                                rec_color = '🟢'
                            elif rec_text in ['Hold', 'Neutral']:
                                rec_color = '🟡'
                            else:
                                rec_color = '🔴'
                            st.metric("Consensus", f"{rec_color} {rec_text}")
                        
                        with col2:
                            num_analysts = forecasts.get('num_analysts', 0)
                            st.metric("Analysts Covering", f"{num_analysts}")
                        
                        with col3:
                            target_mean = forecasts.get('target_mean')
                            if target_mean:
                                st.metric("Avg Target", f"₹{target_mean:,.2f}")
                            else:
                                st.metric("Avg Target", "N/A")
                        
                        with col4:
                            upside = forecasts.get('upside_percent')
                            if upside is not None:
                                delta_color = "normal" if upside >= 0 else "inverse"
                                st.metric("Upside/Downside", f"{upside:+.1f}%", delta_color=delta_color)
                            else:
                                st.metric("Upside/Downside", "N/A")
                        
                        with col5:
                            risk_rating = forecasts.get('risk_rating', 'N/A')
                            st.metric("Risk/Reward", risk_rating)
                        
                        # ========== ROW 2: Target Range ==========
                        if forecasts.get('target_high') and forecasts.get('target_low'):
                            st.markdown("#### 📈 Price Target Range")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Target High", f"₹{forecasts['target_high']:,.2f}")
                            with col2:
                                st.metric("Target Mean", f"₹{forecasts['target_mean']:,.2f}")
                            with col3:
                                st.metric("Target Low", f"₹{forecasts['target_low']:,.2f}")
                            with col4:
                                target_range_pct = forecasts.get('target_range_percent')
                                if target_range_pct:
                                    st.metric("Analyst Spread", f"{target_range_pct:.1f}%")
                                else:
                                    st.metric("Analyst Spread", "N/A")
                        
                        # ========== ROW 3: Earnings Estimates ==========
                        st.markdown("#### 💰 Earnings & Revenue Estimates")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            eps_current = forecasts.get('current_year_eps')
                            if eps_current:
                                st.metric("Current Year EPS", f"₹{eps_current:.2f}")
                            else:
                                st.metric("Current Year EPS", "N/A")
                        
                        with col2:
                            eps_next = forecasts.get('next_year_eps')
                            if eps_next:
                                growth_rate = None
                                if eps_current and eps_current > 0:
                                    growth_rate = ((eps_next - eps_current) / eps_current) * 100
                                st.metric("Next Year EPS", f"₹{eps_next:.2f}", 
                                         f"{growth_rate:+.1f}%" if growth_rate else None)
                            else:
                                st.metric("Next Year EPS", "N/A")
                        
                        with col3:
                            eps_growth = forecasts.get('earnings_growth')
                            if eps_growth:
                                st.metric("EPS Growth Rate", f"{eps_growth*100:.1f}%")
                            else:
                                st.metric("EPS Growth Rate", "N/A")
                        
                        with col4:
                            revenue_growth = forecasts.get('revenue_growth')
                            if revenue_growth:
                                st.metric("Revenue Growth", f"{revenue_growth*100:.1f}%")
                            else:
                                st.metric("Revenue Growth", "N/A")
                        
                        # ========== ROW 4: Valuation Metrics ==========
                        st.markdown("#### 📊 Forward Valuation Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            forward_pe = forecasts.get('forward_pe')
                            if forward_pe:
                                st.metric("Forward P/E", f"{forward_pe:.2f}")
                            else:
                                st.metric("Forward P/E", "N/A")
                        
                        with col2:
                            peg = forecasts.get('peg_ratio')
                            if peg:
                                peg_signal = "Undervalued" if peg < 1 else "Fairly Valued" if peg < 2 else "Overvalued"
                                st.metric("PEG Ratio", f"{peg:.2f}", peg_signal)
                            else:
                                st.metric("PEG Ratio", "N/A")
                        
                        with col3:
                            next_earnings = forecasts.get('next_earnings_date')
                            if next_earnings:
                                st.metric("Next Earnings", str(next_earnings)[:10])
                            else:
                                st.metric("Next Earnings", "N/A")
                        
                        with col4:
                            earnings_est = forecasts.get('earnings_estimate')
                            if earnings_est:
                                st.metric("Est. EPS (Next Qtr)", f"₹{earnings_est:.2f}")
                            else:
                                st.metric("Est. EPS (Next Qtr)", "N/A")
                        
                        # ========== RECOMMENDATION TREND CHART ==========
                        if forecasts.get('recommendation_trend'):
                            st.markdown("#### 📉 Recent Analyst Recommendation Trend")
                            
                            trend_data = forecasts['recommendation_trend']
                            
                            # Create bar chart
                            fig_trend = go.Figure(data=[
                                go.Bar(
                                    x=list(trend_data.keys()),
                                    y=list(trend_data.values()),
                                    marker=dict(
                                        color=['#00cc00' if 'Buy' in k or 'Outperform' in k 
                                               else '#ff9900' if 'Hold' in k or 'Neutral' in k 
                                               else '#ff0000' 
                                               for k in trend_data.keys()]
                                    ),
                                    text=list(trend_data.values()),
                                    textposition='auto'
                                )
                            ])
                            
                            fig_trend.update_layout(
                                title="Distribution of Recent Analyst Recommendations",
                                xaxis_title="Recommendation",
                                yaxis_title="Number of Analysts",
                                height=300,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_trend, use_container_width=True)
                            
                            # Consensus score
                            consensus_score = forecasts.get('consensus_score')
                            if consensus_score:
                                st.markdown(f"""
                                **Consensus Score:** {consensus_score:.2f}/5.0  
                                *(5=Strong Buy, 4=Buy, 3=Hold, 2=Underperform, 1=Sell)*
                                """)
                        
                        # ========== RECENT ANALYST ACTIONS ==========
                        if forecasts.get('recent_recommendations'):
                            with st.expander("📋 Recent Analyst Actions (Last 10)"):
                                recent_recs = forecasts['recent_recommendations']
                                
                                # Create DataFrame
                                rec_df = pd.DataFrame(recent_recs)
                                
                                # Format and display
                                if not rec_df.empty:
                                    # Select relevant columns
                                    display_cols = []
                                    if 'Date' in rec_df.columns or rec_df.index.name:
                                        rec_df['Date'] = rec_df.index if 'Date' not in rec_df.columns else rec_df['Date']
                                        rec_df['Date'] = pd.to_datetime(rec_df['Date']).dt.strftime('%Y-%m-%d')
                                        display_cols.append('Date')
                                    
                                    for col in ['Firm', 'To Grade', 'From Grade', 'Action']:
                                        if col in rec_df.columns:
                                            display_cols.append(col)
                                    
                                    if display_cols:
                                        st.dataframe(
                                            rec_df[display_cols].head(10),
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                else:
                                    st.info("No recent analyst actions available.")
                        
                        # ========== INTERPRETATION & INSIGHTS ==========
                        st.markdown("#### 💡 Analyst Insights")
                        
                        insights = []
                        
                        # Upside insight
                        if upside:
                            if upside > 20:
                                insights.append(f"✅ **Strong Upside**: Analysts see {upside:.1f}% upside potential - significant room for growth")
                            elif upside > 10:
                                insights.append(f"✅ **Moderate Upside**: {upside:.1f}% upside indicates positive outlook")
                            elif upside > 0:
                                insights.append(f"⚠️ **Limited Upside**: Only {upside:.1f}% upside - stock fairly valued")
                            else:
                                insights.append(f"🔴 **Overvalued**: {upside:.1f}% indicates stock may be overpriced")
                        
                        # Consensus insight
                        if rec_text:
                            if rec_text in ['Strong Buy', 'Buy']:
                                insights.append("✅ **Bullish Consensus**: Analysts recommend buying")
                            elif rec_text in ['Hold', 'Neutral']:
                                insights.append("⚠️ **Neutral Stance**: Analysts suggest holding current positions")
                            else:
                                insights.append("🔴 **Bearish Consensus**: Analysts recommend caution or selling")
                        
                        # Growth insight
                        if eps_growth and eps_growth > 0:
                            insights.append(f"📈 **Positive Growth**: {eps_growth*100:.1f}% earnings growth expected")
                        elif eps_growth and eps_growth < 0:
                            insights.append(f"📉 **Negative Growth**: {eps_growth*100:.1f}% earnings decline expected")
                        
                        # PEG insight
                        if peg:
                            if peg < 1:
                                insights.append(f"💰 **Undervalued**: PEG ratio of {peg:.2f} suggests stock is undervalued relative to growth")
                            elif peg > 2:
                                insights.append(f"⚠️ **Expensive**: PEG ratio of {peg:.2f} suggests stock may be overvalued")
                        
                        # Analyst coverage insight
                        if num_analysts:
                            if num_analysts > 15:
                                insights.append(f"👥 **High Coverage**: {num_analysts} analysts covering - well-researched stock")
                            elif num_analysts < 5:
                                insights.append(f"⚠️ **Low Coverage**: Only {num_analysts} analysts - less market consensus")
                        
                        # Display insights
                        if insights:
                            for insight in insights:
                                st.markdown(f"• {insight}")
                        else:
                            st.info("Limited analyst data available for detailed insights.")
                    
                    else:
                        st.info("""
                        **Limited Analyst Coverage**  
                        This stock has limited analyst coverage. This is common for:
                        - Smaller market cap companies
                        - Recently listed stocks
                        - Less liquid securities
                        
                        **Alternative Analysis**: Use technical indicators and fundamental metrics shown above.
                        """)

                except Exception as e:
                    st.error(f"Could not fetch analyst forecasts: {str(e)}")
                    st.info("Try analyzing a large-cap stock like RELIANCE, TCS, or HDFCBANK for comprehensive analyst data.")
                    
                # Trading Signal
                st.markdown('<div class="sub-header">🎯 Trading Signal & Analysis</div>', unsafe_allow_html=True)
                
                overall, signals, score, color = analyzer.get_trading_signal()
                
                signal_cols = st.columns([1, 2])
                with signal_cols[0]:
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                         <h2 style="margin: 0; color: white;">{overall}</h2>
                         <p style="margin: 5px 0; font-size: 18px; color: white;">Score: {score:.1f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge chart for signal strength
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Signal Strength"},
                        gauge={
                            'axis': {'range': [-50, 100]},
                            'bar': {'color': "blue"},
                            'steps': [
                                {'range': [-50, 0], 'color': "red"},
                                {'range': [0, 50], 'color': "yellow"},
                                {'range': [50, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=200, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with signal_cols[1]:
                    with st.expander("📋 Detailed Analysis Signals", expanded=True):
                        for signal in signals:
                            st.markdown(f"• {signal}")
                
                # Pattern Detection
                if show_patterns:
                    st.markdown('<div class="sub-header">📈 Pattern Detection</div>', unsafe_allow_html=True)
                    
                    pattern_tabs = st.tabs(["Dan Zanger Patterns", "Qullamaggie Patterns", "All Patterns"])
                    
                    with pattern_tabs[0]:
                        zanger_patterns = analyzer.detect_chart_patterns()
                        if zanger_patterns:
                            for pattern in zanger_patterns:
                                st.markdown(create_pattern_summary_card(pattern), unsafe_allow_html=True)
                                
                                with st.expander("View Rules & Details"):
                                    st.markdown("**Rules to Follow:**")
                                    for rule in pattern.get('rules', []):
                                        st.markdown(f"• {rule}")
                        else:
                            st.info("No Dan Zanger patterns detected in current timeframe.")
                    
                    with pattern_tabs[1]:
                        swing_patterns = analyzer.detect_swing_patterns()
                        if swing_patterns:
                            for pattern in swing_patterns:
                                st.markdown(create_pattern_summary_card(pattern), unsafe_allow_html=True)
    
                                with st.expander("View Rules & Details"):
                                    st.markdown("**Rules to Follow:**")
                                    for rule in pattern.get('rules', []):
                                        st.markdown(f"• {rule}")
                        else:
                            st.info("No Qullamaggie swing patterns detected in current timeframe.")
                    
                    with pattern_tabs[2]:
                        all_patterns = zanger_patterns + swing_patterns
                        if all_patterns:
                            # Create pattern summary table
                            pattern_df = pd.DataFrame([{
                                'Pattern': p['pattern'],
                                'Signal': p['signal'],
                                'Confidence': p['confidence'],
                                'Score': p['score'],
                                'Description': p['description']
                            } for p in all_patterns])
                            
                            st.dataframe(
                                pattern_df,
                                use_container_width=True,
                                column_config={
                                    "Pattern": st.column_config.TextColumn("Pattern"),
                                    "Signal": st.column_config.TextColumn("Signal"),
                                    "Confidence": st.column_config.TextColumn("Confidence"),
                                    "Score": st.column_config.ProgressColumn(
                                        "Score",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1.0
                                    ),
                                    "Description": st.column_config.TextColumn("Description", width="large")
                                }
                            )
                        else:
                            st.warning("No trading patterns detected.")
                
                # Risk Management
                if show_risk:
                    st.markdown('<div class="sub-header">⚠️ Advanced Risk Management</div>', unsafe_allow_html=True)
                    
                    risk_mgmt = analyzer.get_risk_management(portfolio_value)
                    
                    risk_cols = st.columns(3)
                    
                    with risk_cols[0]:
                        st.markdown("### Entry & Stops")
                        st.metric("Entry Price", f"₹{risk_mgmt['entry_price']:.2f}")
                        
                        # Stop loss levels
                        with st.expander("Stop Loss Levels"):
                            for level_name, stop_price in risk_mgmt['stop_loss_levels'].items():
                                stop_percent = ((risk_mgmt['entry_price'] - stop_price) / risk_mgmt['entry_price']) * 100
                                st.metric(
                                    f"{level_name.title()}",
                                    f"₹{stop_price:.2f}",
                                    f"-{stop_percent:.1f}%"
                                )
                    
                    with risk_cols[1]:
                        st.markdown("### Position Sizing")
                        st.metric("Position Size", f"{risk_mgmt['position_size']:,} shares")
                        st.metric("Position Value", f"₹{risk_mgmt['position_value']:,.2f}")
                        st.metric("Portfolio Risk", f"{risk_mgmt['portfolio_risk_percent']:.2f}%")
                        st.metric("Max Drawdown", f"{risk_mgmt['max_drawdown']:.1f}%")
                        
                    with risk_cols[2]:
                        st.markdown("### Profit Targets")
                        for target_name, target_price in risk_mgmt['profit_targets'].items():
                            if 'target_' in target_name:
                                target_return = ((target_price - risk_mgmt['entry_price']) / risk_mgmt['entry_price']) * 100
                                rr_ratio = risk_mgmt['risk_reward_ratios'].get(target_name, 0)
                                
                                st.metric(
                                    f"{target_name.replace('_', ' ').title()}",
                                    f"₹{target_price:.2f}",
                                    f"+{target_return:.1f}% (R:R 1:{rr_ratio:.1f})"
                                )
                    
                    # Risk/Reward Summary
                    st.markdown("#### Risk/Reward Analysis")
                    rr_df = pd.DataFrame([
                        {
                            'Target': target_name.replace('_', ' ').title(),
                            'Price': f"₹{target_price:.2f}",
                            'Return %': ((target_price - risk_mgmt['entry_price']) / risk_mgmt['entry_price']) * 100,
                            'Risk/Reward': f"1:{rr_ratio:.1f}"
                        }
                        for target_name, target_price in risk_mgmt['profit_targets'].items()
                        if 'target_' in target_name
                        for rr_ratio in [risk_mgmt['risk_reward_ratios'].get(target_name, 0)]
                    ])
                    
                    st.dataframe(rr_df, use_container_width=True, hide_index=True)
                
                # Market Context
                if show_market:
                    st.markdown('<div class="sub-header">🌐 Market Context Analysis</div>', unsafe_allow_html=True)
                    
                    market_context = analyzer.get_market_context()
                    sector_analysis = analyzer.get_sector_analysis()
                    
                    market_cols = st.columns(3)
                    
                    with market_cols[0]:
                        st.markdown("### Nifty 50")
                        if market_context:
                            st.metric("Nifty Level", f"₹{market_context['nifty_level']:.2f}")
                            st.metric("Nifty Trend", market_context['nifty_trend'])
                            st.metric("VIX Level", f"{market_context.get('vix', 'N/A'):.2f}" 
                                     if market_context.get('vix') else 'N/A')
                            st.metric("Market Condition", market_context.get('market_condition', 'N/A'))
                    
                    with market_cols[1]:
                        st.markdown("### Sector Analysis")
                        if sector_analysis:
                            st.metric("Sector", sector_analysis['sector'])
                            st.metric("Stock Return", f"{sector_analysis['stock_return']:.1f}%")
                            st.metric("Sector Return", f"{sector_analysis['sector_return']:.1f}%")
                            st.metric("Relative Strength", sector_analysis['relative_strength'])
                    
                    with market_cols[2]:
                        st.markdown("### Risk Metrics")
                        if sector_analysis:
                            st.metric("Beta", f"{sector_analysis['beta']:.2f}")
                            st.metric("Risk Category", sector_analysis['risk_category'])
                        st.metric("Outperformance", f"{sector_analysis.get('outperformance', 0):.1f}%" 
                                 if sector_analysis else 'N/A')
                
                # Advanced Technical Charts
                if show_advanced:
                    st.markdown('<div class="sub-header">📊 Advanced Technical Charts</div>', unsafe_allow_html=True)
                    
                    chart_tabs = st.tabs(["Price Action & Indicators", "Volume Profile", "Market Structure"])
                    
                    with chart_tabs[0]:
                        fig_candlestick = create_candlestick_chart(analyzer)
                        st.plotly_chart(fig_candlestick, use_container_width=True)
                    
                    with chart_tabs[1]:
                        fig_volume_profile = create_volume_profile_chart(analyzer)
                        st.plotly_chart(fig_volume_profile, use_container_width=True)
                        
                        # Volume Profile Metrics
                        vp = analyzer.detect_volume_profile()
                        if vp:
                            vp_cols = st.columns(4)
                            with vp_cols[0]:
                                st.metric("Point of Control", f"₹{vp['poc_price']:.2f}")
                            with vp_cols[1]:
                                st.metric("Value Area High", f"₹{vp['value_area_high']:.2f}")
                            with vp_cols[2]:
                                st.metric("Value Area Low", f"₹{vp['value_area_low']:.2f}")
                            with vp_cols[3]:
                                st.metric("Value Area %", f"{vp['value_area_percentage']:.1f}%")
                    
                    with chart_tabs[2]:
                        # Create market structure analysis
                        st.info("Market structure analysis shows support/resistance levels and key price zones.")
                        
                        # Simple support/resistance levels
                        df = analyzer.data.tail(100)
                        support_levels = df['Low'].rolling(20).min().dropna().unique()[-3:]
                        resistance_levels = df['High'].rolling(20).max().dropna().unique()[-3:]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Key Support Levels")
                            for i, level in enumerate(sorted(support_levels, reverse=True)[:3], 1):
                                st.metric(f"Support {i}", f"₹{level:.2f}")
                        
                        with col2:
                            st.markdown("#### Key Resistance Levels")
                            for i, level in enumerate(sorted(resistance_levels)[:3], 1):
                                st.metric(f"Resistance {i}", f"₹{level:.2f}")
                
                # Technical Indicators Summary
                st.markdown('<div class="sub-header">📋 Technical Indicators Summary</div>', unsafe_allow_html=True)
                
                current = analyzer.data.iloc[-1]
                
                indicators = {
                    'Trend': [
                        ('SMA 20', f"{current['SMA_20']:.2f}", 'Above' if current['Close'] > current['SMA_20'] else 'Below'),
                        ('SMA 50', f"{current['SMA_50']:.2f}", 'Above' if current['Close'] > current['SMA_50'] else 'Below'),
                        ('SMA 200', f"{current['SMA_200']:.2f}", 'Above' if current['Close'] > current['SMA_200'] else 'Below'),
                        ('EMA Alignment', 'Bullish' if current['EMA_8'] > current['EMA_21'] > current['EMA_55'] else 'Bearish', '')
                    ],
                    'Momentum': [
                        ('RSI 14', f"{current['RSI_14']:.1f}", 
                         'Overbought' if current['RSI_14'] > 70 else 'Oversold' if current['RSI_14'] < 30 else 'Neutral'),
                        ('MACD', f"{current['MACD']:.2f}", 
                         'Bullish' if current['MACD'] > current['MACD_Signal'] else 'Bearish'),
                        ('Stochastic %K', f"{current['Stoch_K']:.1f}", 
                         'Overbought' if current['Stoch_K'] > 80 else 'Oversold' if current['Stoch_K'] < 20 else 'Neutral'),
                        ('Williams %R', f"{current['Williams_%R']:.1f}", 
                         'Oversold' if current['Williams_%R'] < -80 else 'Overbought' if current['Williams_%R'] > -20 else 'Neutral')
                    ],
                    'Volatility': [
                        ('ATR %', f"{(current['ATR_14'] / current['Close']) * 100:.1f}%", 
                         'High' if (current['ATR_14'] / current['Close']) > 0.03 else 'Low'),
                        ('BB Position', f"{(current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower']) * 100:.1f}%",
                         'Upper Band' if current['Close'] > current['BB_Upper'] * 0.95 else 
                         'Lower Band' if current['Close'] < current['BB_Lower'] * 1.05 else 'Middle'),
                        ('Range %', f"{(df['High'].max() - df['Low'].min()) / df['Close'].mean() * 100:.1f}%", '')
                    ],
                    'Volume': [
                        ('Volume Ratio', f"{current['Volume_Ratio']:.1f}x", 
                         'High' if current['Volume_Ratio'] > 1.5 else 'Low'),
                        ('OBV Trend', 'Up' if current['OBV'] > analyzer.data['OBV'].iloc[-20] else 'Down', ''),
                        ('VWAP Diff', f"{(current['Close'] - current['VWAP']) / current['VWAP'] * 100:.1f}%",
                         'Above' if current['Close'] > current['VWAP'] else 'Below')
                    ]
                }
                
                indicator_cols = st.columns(4)
                for idx, (category, metrics) in enumerate(indicators.items()):
                    with indicator_cols[idx]:
                        st.markdown(f"#### {category}")
                        for metric_name, value, interpretation in metrics:
                            st.metric(metric_name, value, interpretation)
                
                # Export Functionality
                st.markdown('<div class="sub-header">📤 Export Analysis</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export to CSV
                    csv_data = export_analysis_report(
                        analyzer, 
                        zanger_patterns + swing_patterns, 
                        risk_mgmt, 
                        (overall, signals, score)
                    )
                    
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_data,
                        file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Generate summary
                    if st.button("📋 Generate Summary", use_container_width=True):
                        summary = f"""
                        ## Analysis Summary for {symbol}
                        
                        **Trading Signal:** {overall} (Score: {score:.1f}/100)
                        
                        **Key Patterns Detected:**
                        {', '.join([p['pattern'] for p in (zanger_patterns + swing_patterns)]) if (zanger_patterns + swing_patterns) else 'None'}
                        
                        **Risk Management:**
                        - Entry: ₹{risk_mgmt['entry_price']:.2f}
                        - Stop Loss: ₹{risk_mgmt['stop_loss']:.2f}
                        - Position Size: {risk_mgmt['position_size']:,} shares
                        - Max Risk: {risk_mgmt['portfolio_risk_percent']:.2f}% of portfolio
                        
                        **Technical Overview:**
                        - Trend: {'Bullish' if current['Close'] > current['SMA_50'] else 'Bearish'}
                        - Momentum: {'Bullish' if current['MACD'] > current['MACD_Signal'] else 'Bearish'}
                        - Volume: {'Strong' if current['Volume'] > current['Volume_SMA_20'] * 1.5 else 'Weak'}
                        
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        """
                        
                        st.success("Summary generated!")
                        st.markdown(summary)
                
                with col3:
                    # Clear cache
                    if st.button("🗑️ Clear Cache", use_container_width=True):
                        st.cache_data.clear()
                        st.rerun()
                
                st.success(f"✅ Analysis completed for {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
            else:
                st.error(f"❌ Unable to fetch data for {symbol}.")
                st.info("💡 Tips:")
                st.markdown("""
                1. Check if the stock symbol is correct
                2. Try adding .NS suffix for NSE stocks (e.g., RELIANCE.NS)
                3. Try adding .BO suffix for BSE stocks (e.g., RELIANCE.BO)
                4. Ensure you have an active internet connection
                5. Try a different analysis period
                """)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h1>Welcome to Indian Equity Market Analyzer Pro</h1>
            <p style="font-size: 18px;">Advanced technical analysis tool based on master trader strategies</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Dan Zanger Strategies</h3>
                <p>Pattern-based trading with volume confirmation</p>
                <ul>
                    <li>Cup and Handle</li>
                    <li>High Tight Flag</li>
                    <li>8% Stop Loss Rule</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Qullamaggie Swing Trading</h3>
                <p>Momentum-based swing trading strategies</p>
                <ul>
                    <li>Episodic Pivots</li>
                    <li>Breakout Patterns</li>
                    <li>1% Risk Management</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Advanced Analytics</h3>
                <p>Comprehensive market analysis tools</p>
                <ul>
                    <li>Volume Profile</li>
                    <li>Risk Management</li>
                    <li>Sector Analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start examples
        st.markdown("### 🚀 Quick Start Examples")
        
        example_cols = st.columns(4)
        examples = [
            ("RELIANCE", "Energy Giant"),
            ("TCS", "IT Services"),
            ("HDFCBANK", "Banking Leader"),
            ("INFY", "IT Major")
        ]
        
        for i, (ex_symbol, ex_desc) in enumerate(examples):
            with example_cols[i]:
                if st.button(f"Analyze {ex_symbol}", use_container_width=True):
                    st.session_state.symbol = ex_symbol
                    st.rerun()
        
        # Features list
        st.markdown("### ✨ Key Features")
        
        features = [
            ("Pattern Detection", "Automated detection of 10+ trading patterns"),
            ("Risk Management", "Advanced position sizing and stop loss calculation"),
            ("Volume Analysis", "Volume profile and institutional flow analysis"),
            ("Market Context", "Sector and broader market analysis"),
            ("Real-time Data", "Live market data with caching"),
            ("Export Reports", "Generate and download analysis reports")
        ]
        
        for feature, description in features:
            st.markdown(f"✅ **{feature}**: {description}")

if __name__ == "__main__":
    main()
