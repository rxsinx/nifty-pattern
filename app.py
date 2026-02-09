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
from pattern_detector import PatternDetector, format_pattern_summary, get_pattern_statistics

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Equity Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
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
    }
    .bullish {
        color: #00ff00;
        font-weight: bold;
    }
    .bearish {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral {
        color: #ffa500;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

class IndianEquityAnalyzer:
    """Master Trader Grade Analysis for Indian Equity Market"""
    
    def __init__(self, symbol, period='1y'):
        """Initialize with NSE/BSE symbol"""
        self.symbol = symbol
        self.period = period
        self.data = None
        self.ticker = None
        self.pattern_detector = None
        
    def fetch_data(self):
        """Fetch data from Yahoo Finance for Indian stocks"""
        try:
            # For NSE stocks, append .NS
            if not self.symbol.endswith('.NS') and not self.symbol.endswith('.BO'):
                ticker_symbol = f"{self.symbol}.NS"
            else:
                ticker_symbol = self.symbol
                
            self.ticker = yf.Ticker(ticker_symbol)
            self.data = self.ticker.history(period=self.period)
            
            if self.data.empty:
                # Try BSE if NSE fails
                ticker_symbol = f"{self.symbol.replace('.NS', '')}.BO"
                self.ticker = yf.Ticker(ticker_symbol)
                self.data = self.ticker.history(period=self.period)
            
            if not self.data.empty:
                self.calculate_indicators()
                # Initialize pattern detector
                self.pattern_detector = PatternDetector(self.data)
                return True
            return False
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return False
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        df = self.data
        
        # Moving Averages
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Low'] = bb.bollinger_lband()
        
        # Stochastic
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume indicators
        df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # VWAP (for intraday analysis)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        self.data = df
    
    def detect_volume_profile(self):
        """Detect volume profile patterns"""
        df = self.data.tail(100)
        
        if len(df) < 20:
            return {
                'poc_price': 0,
                'value_area_high': 0,
                'value_area_low': 0,
                'high_volume_nodes': [],
                'low_volume_nodes': [],
                'volume_distribution': np.array([]),
                'price_bins': np.array([])
            }
        
        # Calculate price levels and volume distribution
        price_range = df['High'].max() - df['Low'].min()
        num_bins = 50
        bins = np.linspace(df['Low'].min(), df['High'].max(), num_bins)
        
        volume_at_price = []
        for i in range(len(bins) - 1):
            mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
            volume_at_price.append(df[mask]['Volume'].sum())
        
        volume_at_price = np.array(volume_at_price)
        
        if len(volume_at_price) == 0 or volume_at_price.sum() == 0:
            return {
                'poc_price': df['Close'].iloc[-1],
                'value_area_high': df['High'].max(),
                'value_area_low': df['Low'].min(),
                'high_volume_nodes': [],
                'low_volume_nodes': [],
                'volume_distribution': volume_at_price,
                'price_bins': bins
            }
        
        # Find Point of Control (POC) - highest volume price level
        poc_idx = np.argmax(volume_at_price)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
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
        
        if value_area_indices:
            value_area_high = bins[max(value_area_indices) + 1]
            value_area_low = bins[min(value_area_indices)]
        else:
            value_area_high = df['High'].max()
            value_area_low = df['Low'].min()
        
        # Identify high and low volume nodes
        threshold_high = np.percentile(volume_at_price[volume_at_price > 0], 80) if np.any(volume_at_price > 0) else 0
        threshold_low = np.percentile(volume_at_price[volume_at_price > 0], 20) if np.any(volume_at_price > 0) else 0
        
        high_volume_nodes = bins[:-1][volume_at_price > threshold_high]
        low_volume_nodes = bins[:-1][volume_at_price < threshold_low]
        
        return {
            'poc_price': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'high_volume_nodes': high_volume_nodes,
            'low_volume_nodes': low_volume_nodes,
            'volume_distribution': volume_at_price,
            'price_bins': bins
        }
    
    def detect_chart_patterns(self):
        """Detect Dan Zanger's Chart Patterns using PatternDetector"""
        if not self.pattern_detector:
            return []
        
        try:
            patterns = self.pattern_detector.detect_all_zanger_patterns()
            return patterns
        except Exception as e:
            st.warning(f"Pattern detection error: {e}")
            return []
    
    def detect_swing_patterns(self):
        """Detect Qullamaggie's Swing Patterns using PatternDetector"""
        if not self.pattern_detector:
            return []
        
        try:
            patterns = self.pattern_detector.detect_all_swing_patterns()
            return patterns
        except Exception as e:
            st.warning(f"Swing pattern detection error: {e}")
            return []
    
    def get_trading_signal(self):
        """Generate comprehensive trading signal"""
        df = self.data
        current = df.iloc[-1]
        
        signals = []
        score = 0
        
        # Trend Analysis
        if current['Close'] > current['SMA_20']:
            signals.append("✅ Price above 20 SMA (Short-term bullish)")
            score += 1
        else:
            signals.append("❌ Price below 20 SMA (Short-term bearish)")
            score -= 1
        
        if current['Close'] > current['SMA_50']:
            signals.append("✅ Price above 50 SMA (Medium-term bullish)")
            score += 1
        else:
            signals.append("❌ Price below 50 SMA (Medium-term bearish)")
            score -= 1
        
        if current['Close'] > current['SMA_200']:
            signals.append("✅ Price above 200 SMA (Long-term bullish)")
            score += 2
        else:
            signals.append("❌ Price below 200 SMA (Long-term bearish)")
            score -= 2
        
        # MACD
        if current['MACD'] > current['MACD_Signal']:
            signals.append("✅ MACD bullish crossover")
            score += 1
        else:
            signals.append("❌ MACD bearish crossover")
            score -= 1
        
        # RSI
        if current['RSI'] > 70:
            signals.append("⚠️ RSI Overbought (>70)")
            score -= 1
        elif current['RSI'] < 30:
            signals.append("✅ RSI Oversold (<30) - Potential reversal")
            score += 1
        else:
            signals.append(f"✅ RSI Neutral ({current['RSI']:.2f})")
        
        # Volume Analysis
        if current['Volume'] > current['Volume_SMA']:
            signals.append("✅ Above average volume (Strong interest)")
            score += 1
        else:
            signals.append("⚠️ Below average volume (Weak interest)")
        
        # Determine overall signal
        if score >= 4:
            overall = "🟢 STRONG BUY"
        elif score >= 2:
            overall = "🟢 BUY"
        elif score >= -1:
            overall = "🟡 HOLD"
        elif score >= -3:
            overall = "🔴 SELL"
        else:
            overall = "🔴 STRONG SELL"
        
        return overall, signals, score
    
    def get_risk_management(self):
        """Calculate risk management parameters based on Zanger/Qullamaggie rules"""
        df = self.data
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Stop loss based on ATR and breakout point
        stop_loss_atr = current_price - (2 * atr)  # 2 ATR stop
        stop_loss_percent = current_price * 0.98  # 2% hard stop
        
        stop_loss = max(stop_loss_atr, stop_loss_percent)
        
        # Profit targets
        target_1 = current_price * 1.15  # 15% gain (Zanger: 20-30%)
        target_2 = current_price * 1.30  # 30% gain
        
        # Position sizing (1% portfolio risk rule)
        risk_per_share = current_price - stop_loss
        
        return {
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'risk_per_share': risk_per_share,
            'risk_reward_1': (target_1 - current_price) / risk_per_share if risk_per_share > 0 else 0,
            'risk_reward_2': (target_2 - current_price) / risk_per_share if risk_per_share > 0 else 0
        }
    
    def get_company_info(self):
        """Get company fundamental information"""
        info = {}
        try:
            ticker_info = self.ticker.info
            info = {
                'name': ticker_info.get('longName', 'N/A'),
                'sector': ticker_info.get('sector', 'N/A'),
                'industry': ticker_info.get('industry', 'N/A'),
                'market_cap': ticker_info.get('marketCap', 0),
                'pe_ratio': ticker_info.get('trailingPE', 'N/A'),
                'pb_ratio': ticker_info.get('priceToBook', 'N/A'),
                'dividend_yield': ticker_info.get('dividendYield', 0),
                'eps': ticker_info.get('trailingEps', 'N/A'),
                '52w_high': ticker_info.get('fiftyTwoWeekHigh', 0),
                '52w_low': ticker_info.get('fiftyTwoWeekLow', 0),
                'beta': ticker_info.get('beta', 'N/A'),
                'description': ticker_info.get('longBusinessSummary', 'N/A')
            }
        except:
            pass
        
        return info

def extract_price_from_string(price_str):
    """Extract numeric price from string like '₹1234.56'"""
    if isinstance(price_str, (int, float)):
        return float(price_str)
    try:
        # Remove currency symbol and extract number
        import re
        numbers = re.findall(r'[\d.]+', str(price_str))
        if numbers:
            return float(numbers[0])
    except:
        pass
    return None

def draw_patterns_on_chart(fig, patterns, df):
    """
    Draw detected patterns on the chart with entry/exit points.
    
    Args:
        fig: Plotly figure object
        patterns: List of detected patterns
        df: Price data DataFrame
    """
    if not patterns:
        return fig
    
    # Get the last date for annotations
    last_date = df.index[-1]
    
    for i, pattern in enumerate(patterns):
        # Extract prices from pattern
        entry_price = extract_price_from_string(pattern.get('entry_point', ''))
        stop_loss = extract_price_from_string(pattern.get('stop_loss', ''))
        target_1 = extract_price_from_string(pattern.get('target_1', ''))
        target_2 = extract_price_from_string(pattern.get('target_2', ''))
        
        pattern_name = pattern.get('pattern', 'Unknown')
        signal = pattern.get('signal', 'NEUTRAL')
        
        # Choose color based on signal
        if signal == 'BULLISH':
            color = '#00cc66'  # Green
        elif signal == 'BEARISH':
            color = '#ff4d4d'  # Red
        else:
            color = '#ffa500'  # Orange
        
        # Draw Entry Point
        if entry_price:
            fig.add_hline(
                y=entry_price,
                line_dash="dash",
                line_color=color,
                line_width=2,
                annotation_text=f"📍 ENTRY: ₹{entry_price:.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Draw Stop Loss
        if stop_loss:
            fig.add_hline(
                y=stop_loss,
                line_dash="dot",
                line_color="red",
                line_width=2,
                annotation_text=f"🛑 STOP: ₹{stop_loss:.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Draw Target 1
        if target_1:
            fig.add_hline(
                y=target_1,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"🎯 T1: ₹{target_1:.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Draw Target 2
        if target_2:
            fig.add_hline(
                y=target_2,
                line_dash="dot",
                line_color="darkgreen",
                line_width=2,
                annotation_text=f"🎯 T2: ₹{target_2:.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Add pattern label annotation
        fig.add_annotation(
            x=last_date,
            y=df['High'].max() * (1 - 0.05 * i),  # Stack annotations
            text=f"<b>{pattern_name}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=-50,
            ay=-30,
            bgcolor=color,
            font=dict(color='white', size=12),
            bordercolor=color,
            borderwidth=2,
            borderpad=4,
            opacity=0.9,
            row=1, col=1
        )
    
    return fig

def create_candlestick_chart(analyzer, patterns=None):
    """Create advanced candlestick chart with indicators and patterns"""
    df = analyzer.data.tail(100)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('Price Action with Indicators & Patterns', 'MACD', 'RSI', 'Volume Profile')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], name='EMA 10', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', width=1, dash='dot')), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red', width=1)), row=2, col=1)
    
    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color=colors), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    colors_vol = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_SMA'], name='Vol SMA', line=dict(color='orange', width=2)), row=4, col=1)
    
    # Draw patterns if provided
    if patterns:
        fig = draw_patterns_on_chart(fig, patterns, df)
    
    # Update layout
    fig.update_layout(
        title=f'{analyzer.symbol} - Master Trader Analysis with Pattern Detection',
        xaxis_rangeslider_visible=False,
        height=1200,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def create_volume_profile_chart(analyzer):
    """Create volume profile chart"""
    vp = analyzer.detect_volume_profile()
    
    fig = go.Figure()
    
    if len(vp['volume_distribution']) == 0:
        fig.add_annotation(text="Insufficient data for volume profile", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Horizontal volume bars
    price_levels = (vp['price_bins'][:-1] + vp['price_bins'][1:]) / 2
    
    fig.add_trace(go.Bar(
        y=price_levels,
        x=vp['volume_distribution'],
        orientation='h',
        name='Volume at Price',
        marker=dict(color='lightblue', line=dict(color='blue', width=1))
    ))
    
    # Point of Control
    fig.add_hline(y=vp['poc_price'], line_dash="solid", line_color="black", 
                  annotation_text="POC", line_width=3)
    
    # Value Area
    fig.add_hrect(y0=vp['value_area_low'], y1=vp['value_area_high'], 
                  line_width=0, fillcolor="red", opacity=0.2,
                  annotation_text="Value Area", annotation_position="right")
    
    # High Volume Nodes
    for hvn in vp['high_volume_nodes'][:3]:
        fig.add_hline(y=hvn, line_dash="dash", line_color="green", 
                      annotation_text="HVN", line_width=1)
    
    # Low Volume Nodes
    for lvn in vp['low_volume_nodes'][:3]:
        fig.add_hline(y=lvn, line_dash="dash", line_color="orange", 
                      annotation_text="LVN", line_width=1)
    
    fig.update_layout(
        title='Volume Profile Analysis',
        xaxis_title='Volume',
        yaxis_title='Price',
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    st.markdown('<div class="main-header">🎯 Indian Equity Market Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Master Trader Grade Analysis - Dan Zanger & Qullamaggie Strategies</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Settings")
        
        symbol = st.text_input(
            "Enter Stock Symbol (NSE)",
            value="RELIANCE",
            help="Enter NSE symbol without .NS suffix (e.g., RELIANCE, TCS, INFY)"
        )
        
        period = st.selectbox(
            "Analysis Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3
        )
        
        show_patterns_on_chart = st.checkbox("Show Patterns on Chart", value=True, 
                                             help="Draw pattern lines and entry/exit points on the chart")
        
        analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📚 Pattern Detection")
        st.markdown("""
        **Dan Zanger (6):**
        - ✅ Cup and Handle
        - ✅ High Tight Flag
        - ✅ Ascending Triangle
        - ✅ Flat Base
        - ✅ Falling Wedge
        - ✅ Double Bottom
        
        **Classic (8):**
        - ✅ Head & Shoulders 🔻
        - ✅ Double Top 🔻
        - ✅ Descending Triangle 🔻
        - ✅ Symmetrical Triangle
        - ✅ Bull Flag
        - ✅ Bear Flag 🔻
        - ✅ Rising Wedge 🔻
        - ✅ Pennant (Bull/Bear)
        
        **Qullamaggie (5):**
        - ✅ Episodic Pivot
        - ✅ Breakout
        - ✅ Parabolic Short 🔻
        - ✅ Gap and Go
        - ✅ ABCD Pattern
        
        🔻 = **SHORT Signal**
        """)
        
        st.markdown("---")
        st.markdown("### 🎓 Trading Philosophy")
        st.markdown("""
        **Dan Zanger's Rules:**
        - Volume is everything
        - Focus on liquid leaders
        - 8% absolute sell rule
        
        **Qullamaggie's Rules:**
        - Extreme discipline
        - Market leaders only
        - Never risk >1% per trade
        """)
    
    if analyze_btn:
        with st.spinner(f'🔄 Analyzing {symbol}...'):
            analyzer = IndianEquityAnalyzer(symbol, period)
            
            if analyzer.fetch_data():
                # Detect all patterns first
                zanger_patterns = analyzer.detect_chart_patterns()
                swing_patterns = analyzer.detect_swing_patterns()
                classic_patterns = analyzer.pattern_detector.detect_all_classic_patterns() if analyzer.pattern_detector else []
                all_patterns = zanger_patterns + swing_patterns + classic_patterns
                
                # Company Information
                st.markdown('<div class="sub-header">🏢 Company Overview</div>', unsafe_allow_html=True)
                
                info = analyzer.get_company_info()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Company", info.get('name', 'N/A'))
                    st.metric("Sector", info.get('sector', 'N/A'))
                with col2:
                    market_cap = info.get('market_cap', 0)
                    st.metric("Market Cap", f"₹{market_cap/10000000:.2f} Cr" if market_cap else 'N/A')
                    st.metric("Industry", info.get('industry', 'N/A'))
                with col3:
                    st.metric("P/E Ratio", f"{info.get('pe_ratio', 'N/A'):.2f}" if isinstance(info.get('pe_ratio'), (int, float)) else 'N/A')
                    st.metric("P/B Ratio", f"{info.get('pb_ratio', 'N/A'):.2f}" if isinstance(info.get('pb_ratio'), (int, float)) else 'N/A')
                with col4:
                    st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else 'N/A')
                    div_yield = info.get('dividend_yield', 0)
                    st.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else 'N/A')
                
                # Current Price Information
                current = analyzer.data.iloc[-1]
                prev = analyzer.data.iloc[-2]
                change = current['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Current Price", f"₹{current['Close']:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                with col2:
                    st.metric("Day High", f"₹{current['High']:.2f}")
                with col3:
                    st.metric("Day Low", f"₹{current['Low']:.2f}")
                with col4:
                    st.metric("52W High", f"₹{info.get('52w_high', 0):.2f}")
                with col5:
                    st.metric("52W Low", f"₹{info.get('52w_low', 0):.2f}")
                
                # Trading Signal
                st.markdown('<div class="sub-header">🎯 Trading Signal</div>', unsafe_allow_html=True)
                
                overall, signals, score = analyzer.get_trading_signal()
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"### {overall}")
                    st.markdown(f"**Signal Strength: {score}/7**")
                with col2:
                    for signal in signals:
                        st.markdown(signal)
                
                # Risk Management
                st.markdown('<div class="sub-header">⚠️ Risk Management Parameters</div>', unsafe_allow_html=True)
                
                risk_mgmt = analyzer.get_risk_management()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Entry & Stop Loss**")
                    st.metric("Entry Price", f"₹{risk_mgmt['entry_price']:.2f}")
                    st.metric("Stop Loss", f"₹{risk_mgmt['stop_loss']:.2f}", 
                             f"-{((risk_mgmt['entry_price'] - risk_mgmt['stop_loss'])/risk_mgmt['entry_price']*100):.2f}%")
                
                with col2:
                    st.markdown("**Profit Targets**")
                    st.metric("Target 1 (15%)", f"₹{risk_mgmt['target_1']:.2f}")
                    st.metric("Target 2 (30%)", f"₹{risk_mgmt['target_2']:.2f}")
                
                with col3:
                    st.markdown("**Risk/Reward**")
                    st.metric("R:R Target 1", f"1:{risk_mgmt['risk_reward_1']:.2f}")
                    st.metric("R:R Target 2", f"1:{risk_mgmt['risk_reward_2']:.2f}")
                    st.metric("Risk/Share", f"₹{risk_mgmt['risk_per_share']:.2f}")
                
                # Chart Patterns
                st.markdown('<div class="sub-header">📈 Chart Pattern Detection</div>', unsafe_allow_html=True)
                
                tab1, tab2, tab3 = st.tabs(["Dan Zanger Patterns", "Qullamaggie Patterns", "Classic Patterns"])
                
                with tab1:
                    if zanger_patterns:
                        st.success(f"✅ Found {len(zanger_patterns)} Dan Zanger pattern(s)")
                        for pattern in zanger_patterns:
                            with st.expander(f"🔹 {pattern['pattern']} - {pattern['signal']}", expanded=True):
                                st.markdown(f"**Description:** {pattern['description']}")
                                st.markdown(f"**Action:** {pattern['action']}")
                                
                                # Display entry/exit points
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    if 'entry_point' in pattern:
                                        st.markdown(f"**📍 Entry:** {pattern['entry_point']}")
                                with col2:
                                    if 'stop_loss' in pattern:
                                        st.markdown(f"**🛑 Stop:** {pattern['stop_loss']}")
                                with col3:
                                    if 'target_1' in pattern:
                                        st.markdown(f"**🎯 T1:** {pattern['target_1']}")
                                with col4:
                                    if 'target_2' in pattern:
                                        st.markdown(f"**🎯 T2:** {pattern['target_2']}")
                    else:
                        st.info("No Dan Zanger patterns detected in current timeframe")
                
                with tab2:
                    if swing_patterns:
                        st.success(f"✅ Found {len(swing_patterns)} Qullamaggie pattern(s)")
                        for pattern in swing_patterns:
                            with st.expander(f"🔹 {pattern['pattern']} - {pattern['signal']}", expanded=True):
                                st.markdown(f"**Description:** {pattern['description']}")
                                st.markdown(f"**Action:** {pattern['action']}")
                                
                                # Display entry/exit points
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    if 'entry_point' in pattern:
                                        st.markdown(f"**📍 Entry:** {pattern.get('entry_point', 'N/A')}")
                                with col2:
                                    if 'stop_loss' in pattern:
                                        st.markdown(f"**🛑 Stop:** {pattern.get('stop_loss', 'N/A')}")
                                with col3:
                                    if 'target_1' in pattern:
                                        st.markdown(f"**🎯 T1:** {pattern.get('target_1', 'N/A')}")
                                with col4:
                                    if 'target_2' in pattern:
                                        st.markdown(f"**🎯 T2:** {pattern.get('target_2', 'N/A')}")
                    else:
                        st.info("No Qullamaggie swing patterns detected in current timeframe")
                
                with tab3:
                    if classic_patterns:
                        st.success(f"✅ Found {len(classic_patterns)} Classic pattern(s)")
                        
                        # Separate bullish and bearish patterns
                        bullish_classic = [p for p in classic_patterns if p.get('signal') == 'BULLISH']
                        bearish_classic = [p for p in classic_patterns if p.get('signal') == 'BEARISH']
                        neutral_classic = [p for p in classic_patterns if p.get('signal') not in ['BULLISH', 'BEARISH']]
                        
                        if bullish_classic:
                            st.markdown("### 🟢 Bullish Patterns")
                            for pattern in bullish_classic:
                                with st.expander(f"🔹 {pattern['pattern']} - {pattern['signal']}", expanded=True):
                                    st.markdown(f"**Description:** {pattern['description']}")
                                    st.markdown(f"**Action:** {pattern['action']}")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        if 'entry_point' in pattern:
                                            st.markdown(f"**📍 Entry:** {pattern.get('entry_point', 'N/A')}")
                                    with col2:
                                        if 'stop_loss' in pattern:
                                            st.markdown(f"**🛑 Stop:** {pattern.get('stop_loss', 'N/A')}")
                                    with col3:
                                        if 'target_1' in pattern:
                                            st.markdown(f"**🎯 T1:** {pattern.get('target_1', 'N/A')}")
                                    with col4:
                                        if 'target_2' in pattern:
                                            st.markdown(f"**🎯 T2:** {pattern.get('target_2', 'N/A')}")
                        
                        if bearish_classic:
                            st.markdown("### 🔴 Bearish Patterns (SHORT Opportunities)")
                            for pattern in bearish_classic:
                                with st.expander(f"🔻 {pattern['pattern']} - {pattern['signal']}", expanded=True):
                                    st.markdown(f"**Description:** {pattern['description']}")
                                    st.markdown(f"**Action:** {pattern['action']}")
                                    st.warning("⚠️ **SHORT POSITION** - Profit from declining prices")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        if 'entry_point' in pattern:
                                            st.markdown(f"**📍 SHORT Entry:** {pattern.get('entry_point', 'N/A')}")
                                    with col2:
                                        if 'stop_loss' in pattern:
                                            st.markdown(f"**🛑 Stop:** {pattern.get('stop_loss', 'N/A')}")
                                    with col3:
                                        if 'target_1' in pattern:
                                            st.markdown(f"**🎯 T1:** {pattern.get('target_1', 'N/A')}")
                                    with col4:
                                        if 'target_2' in pattern:
                                            st.markdown(f"**🎯 T2:** {pattern.get('target_2', 'N/A')}")
                        
                        if neutral_classic:
                            st.markdown("### ⚡ Neutral/Breakout Patterns")
                            for pattern in neutral_classic:
                                with st.expander(f"⚡ {pattern['pattern']} - {pattern['signal']}", expanded=True):
                                    st.markdown(f"**Description:** {pattern['description']}")
                                    st.markdown(f"**Action:** {pattern['action']}")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        if 'entry_point' in pattern:
                                            st.markdown(f"**📍 Entry:** {pattern.get('entry_point', 'N/A')}")
                                    with col2:
                                        if 'stop_loss' in pattern:
                                            st.markdown(f"**🛑 Stop:** {pattern.get('stop_loss', 'N/A')}")
                                    with col3:
                                        if 'target_1' in pattern:
                                            st.markdown(f"**🎯 T1:** {pattern.get('target_1', 'N/A')}")
                                    with col4:
                                        if 'target_2' in pattern:
                                            st.markdown(f"**🎯 T2:** {pattern.get('target_2', 'N/A')}")
                    else:
                        st.info("No Classic patterns detected in current timeframe")
                
                # Charts
                st.markdown('<div class="sub-header">📊 Technical Analysis Charts</div>', unsafe_allow_html=True)
                
                chart_tab1, chart_tab2 = st.tabs(["Price Action & Indicators", "Volume Profile"])
                
                with chart_tab1:
                    # Create chart with or without patterns based on user preference
                    if show_patterns_on_chart and all_patterns:
                        st.info(f"📌 Showing {len(all_patterns)} detected pattern(s) on chart")
                        fig_candlestick = create_candlestick_chart(analyzer, all_patterns)
                    else:
                        fig_candlestick = create_candlestick_chart(analyzer)
                    
                    st.plotly_chart(fig_candlestick, use_container_width=True)
                
                with chart_tab2:
                    fig_volume_profile = create_volume_profile_chart(analyzer)
                    st.plotly_chart(fig_volume_profile, use_container_width=True)
                    
                    # Volume Profile Metrics
                    vp = analyzer.detect_volume_profile()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Point of Control (POC)", f"₹{vp['poc_price']:.2f}")
                    with col2:
                        st.metric("Value Area High", f"₹{vp['value_area_high']:.2f}")
                    with col3:
                        st.metric("Value Area Low", f"₹{vp['value_area_low']:.2f}")
                
                # Key Metrics Table
                st.markdown('<div class="sub-header">📋 Key Technical Indicators</div>', unsafe_allow_html=True)
                
                metrics_df = pd.DataFrame({
                    'Indicator': ['RSI', 'MACD', 'Signal Line', 'Stochastic %K', 'Stochastic %D', 'ATR', 'OBV'],
                    'Current Value': [
                        f"{current['RSI']:.2f}",
                        f"{current['MACD']:.2f}",
                        f"{current['MACD_Signal']:.2f}",
                        f"{current['Stoch_K']:.2f}",
                        f"{current['Stoch_D']:.2f}",
                        f"{current['ATR']:.2f}",
                        f"{current['OBV']:.0f}"
                    ],
                    'Interpretation': [
                        'Overbought' if current['RSI'] > 70 else 'Oversold' if current['RSI'] < 30 else 'Neutral',
                        'Bullish' if current['MACD'] > current['MACD_Signal'] else 'Bearish',
                        '-',
                        'Overbought' if current['Stoch_K'] > 80 else 'Oversold' if current['Stoch_K'] < 20 else 'Neutral',
                        'Overbought' if current['Stoch_D'] > 80 else 'Oversold' if current['Stoch_D'] < 20 else 'Neutral',
                        'High Volatility' if current['ATR'] > analyzer.data['ATR'].mean() * 1.5 else 'Normal',
                        'Accumulation' if current['OBV'] > analyzer.data['OBV'].mean() else 'Distribution'
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Trading Rules Summary
                st.markdown('<div class="sub-header">📖 Master Trader Rules Summary</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ### 🎯 Dan Zanger's Golden Rules
                    1. **Volume is Everything** - Pattern must break with 3x average volume
                    2. **8% Absolute Sell Rule** - Cut losses without emotion
                    3. **Focus on Liquid Leaders** - Trade stocks in strong sectors
                    4. **Patience Pays** - Wait 7-8 weeks for cup formation
                    5. **Upper Half Entry** - Handle must be in upper half of cup
                    6. **Pure Technicals** - Price & volume tell the story
                    """)
                
                with col2:
                    st.markdown("""
                    ### 🎓 Qullamaggie's Swing Rules
                    1. **Extreme Discipline** - Rigid adherence prevents emotional mistakes
                    2. **1% Risk Rule** - Never risk more than 1% of portfolio
                    3. **Market Leaders Only** - Focus on strongest stocks in strongest groups
                    4. **ORH Entry** - Opening Range High entry for episodic pivots
                    5. **VDU = Gold** - Volume Dry Up shows selling exhaustion
                    6. **Momentum Trading** - Follow institutional money flow
                    7. **3-5 Day Hold** - Quick profits, trail winners with 10/20 SMA
                    """)
                
                st.success(f"✅ Analysis completed for {symbol}")
                
            else:
                st.error(f"❌ Unable to fetch data for {symbol}. Please check the symbol and try again.")
                st.info("💡 Tip: Try adding .NS for NSE stocks or .BO for BSE stocks")

if __name__ == "__main__":
    main()
