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
from quantitative_analysis import FractalAnalysis, StatisticalEstimation, VolatilityModelling, run_full_quantitative_analysis

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
        df['SMA_21'] = SMAIndicator(df['Close'], window=21).sma_indicator()
        df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['EMA_70'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        #df['RSI_MA14'] = RSIIndicator(df['Close'], window=14).rsi()
        
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
        df = self.data.tail(200)
        
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
        
        # Special handling for Darvas Box - draw the box
        if 'Darvas Box' in pattern_name and 'box_data' in pattern:
            box_data = pattern['box_data']
            box_top = box_data['top']
            box_bottom = box_data['bottom']
            box_start = box_data['start_date']
            box_end = box_data['end_date']
            
            # Draw box rectangle
            fig.add_shape(
                type="rect",
                x0=box_start,
                x1=box_end,
                y0=box_bottom,
                y1=box_top,
                line=dict(color='#4169E1', width=3, dash='dash'),
                fillcolor='rgba(65, 105, 225, 0.1)',
                row=1, col=1
            )
            
            # Label box top and bottom
            fig.add_hline(
                y=box_top,
                line_dash="solid",
                line_color='#4169E1',
                line_width=1,
                annotation_text=f"📦 Box Top: ₹{box_top:.2f}",
                annotation_position="right",
                row=1, col=1
            )
            
            fig.add_hline(
                y=box_bottom,
                line_dash="solid",
                line_color='#4169E1',
                line_width=1,
                annotation_text=f"📦 Box Bottom: ₹{box_bottom:.2f}",
                annotation_position="right",
                row=1, col=1
            )
            
            # Add "DARVAS BOX" label - DELETED
            
            
        
        # Special handling for Order Blocks - draw the zone
        if 'Order Block' in pattern_name and 'order_block_data' in pattern:
            ob_data = pattern['order_block_data']
            ob_high = ob_data['high']
            ob_low = ob_data['low']
            ob_type = ob_data['type']
            
            # Calculate zone start (show last 30% of chart)
            zone_start = df.index[int(len(df) * 0.7)]
            zone_end = df.index[-1]
            
            # Draw order block zone
            ob_color = 'rgba(0, 255, 0, 0.15)' if ob_type == 'BULLISH' else 'rgba(255, 0, 0, 0.15)'
            border_color = '#00cc66' if ob_type == 'BULLISH' else '#ff4d4d'
            
            fig.add_shape(
                type="rect",
                x0=zone_start,
                x1=zone_end,
                y0=ob_low,
                y1=ob_high,
                line=dict(color=border_color, width=2, dash='dot'),
                fillcolor=ob_color,
                row=1, col=1
            )
            
            # Add order block label
            fig.add_annotation(
                x=zone_end,
                y=(ob_high + ob_low) / 2,
                text=f"<b>OB</b>",
                showarrow=False,
                font=dict(color=border_color, size=12, family='Arial Black'),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=border_color,
                borderwidth=2,
                xanchor='left',
                row=1, col=1
            )
        
        # Special handling for Elliott Wave - draw correction levels
        if 'Elliott Wave' in pattern_name and 'elliott_data' in pattern:
            ew_data = pattern['elliott_data']
            if 'correction_levels' in ew_data:
                levels = ew_data['correction_levels']
                
                # Draw Fibonacci correction levels
                colors_map = {
                    '38.2%': '#FFD700',  # Gold
                    '50%': '#FF8C00',    # Dark Orange
                    '61.8%': '#FF4500'   # Orange Red
                }
                
                for level_name, level_price in levels.items():
                    fig.add_hline(
                        y=level_price,
                        line_dash="dot",
                        line_color=colors_map.get(level_name, '#FFA500'),
                        line_width=1,
                        annotation_text=f"Wave C {level_name}: ₹{level_price:.2f}",
                        annotation_position="left",
                        row=1, col=1
                    )
        
        # Special handling for Mean Reversion - draw Bollinger Bands and mean
        if 'Mean Reversion' in pattern_name and 'mean_reversion_data' in pattern:
            mr_data = pattern['mean_reversion_data']
            bb_upper = mr_data['bb_upper']
            bb_mid = mr_data['bb_mid']
            bb_lower = mr_data['bb_lower']
            std_dev = mr_data['std_deviation']
            
            # Draw Bollinger Bands
            fig.add_hline(
                y=bb_upper,
                line_dash="dash",
                line_color='#FF6B6B',  # Red
                line_width=2,
                annotation_text=f"Upper BB: ₹{bb_upper:.2f} (+2σ)",
                annotation_position="left",
                row=1, col=1
            )
            
            fig.add_hline(
                y=bb_mid,
                line_dash="solid",
                line_color='#4ECDC4',  # Teal (Mean)
                line_width=3,
                annotation_text=f"MEAN (SMA-20): ₹{bb_mid:.2f}",
                annotation_position="left",
                row=1, col=1
            )
            
            fig.add_hline(
                y=bb_lower,
                line_dash="dash",
                line_color='#95E1D3',  # Light Green
                line_width=2,
                annotation_text=f"Lower BB: ₹{bb_lower:.2f} (-2σ)",
                annotation_position="left",
                row=1, col=1
            )
            
            # Add standard deviation annotation
            current_price = df['Close'].iloc[-1]
            annotation_y = current_price * 1.02
            
            fig.add_annotation(
                x=df.index[-5] if len(df) > 5 else df.index[-1],
                y=annotation_y,
                text=f"<b>{abs(std_dev):.1f}σ from mean</b>",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#FF6B6B' if std_dev > 0 else '#95E1D3',
                ax=0,
                ay=-40,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#FF6B6B' if std_dev > 0 else '#95E1D3',
                borderwidth=2,
                font=dict(size=12, color='black'),
                row=1, col=1
            )
        
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
        
        # Add pattern label annotation (skip for special patterns with custom labels)
        if 'Darvas Box' not in pattern_name and 'Order Block' not in pattern_name:
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
    df = analyzer.data.tail(200)
    
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
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_21'], name='SMA 21', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], name='EMA 10', line=dict(color='green', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_70'], name='EMA 70', line=dict(color='black', width=1.5, dash='dash')), row=1, col=1)
    
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
    #fig.add_trace(go.Scatter(x=df.index, y=df['RSI_MA14'], name='RSI14', line=dict(color='black', width=2)), row=3, col=1)
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
    st.markdown('<p style="text-align: center; color: gray;">Master Trader Grade Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Settings")
        
        symbol = st.text_input(
            "Enter Stock Symbol (NSE)",
            value="",
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
        
        **Advanced (11):**
        - ✅ VCP 🔥 (Minervini)
        - ✅ Darvas Box 📦
        - ✅ Wyckoff Accumulation 📊
        - ✅ Wyckoff Distribution 🔻
        - ✅ CANSLIM Setup 💎
        - ✅ Inv H&S 🔄 (Bullish Rev)
        - ✅ Triple Top 🔻🔻🔻
        - ✅ Triple Bottom 💚💚💚
        - ✅ Order Blocks 🏦 (ICT)
        - ✅ Elliott Wave 🌊 (5-3)
        - ✅ Mean Reversion 📉📈 (Stats)
        
        **TOTAL: 30 Patterns**
        
        🔻 = **SHORT Signal**
        📊 = **Smart Money**
        💎 = **Growth Stock**
        🔥 = **Stage 2 Uptrend**
        📦 = **Box Breakout**
        🔄 = **Bullish Reversal**
        🏦 = **Institutional Zones**
        🌊 = **Wave Theory**
        📉📈 = **Statistical Edge**
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
                wyckoff_canslim_patterns = analyzer.pattern_detector.detect_all_wyckoff_canslim_patterns() if analyzer.pattern_detector else []
                all_patterns = zanger_patterns + swing_patterns + classic_patterns + wyckoff_canslim_patterns
                
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
                
                tab1, tab2, tab3, tab4 = st.tabs(["Dan Zanger Patterns", "Qullamaggie Patterns", "Classic Patterns", "Advanced Patterns"])
                
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
                
                with tab4:
                    if wyckoff_canslim_patterns:
                        st.success(f"✅ Found {len(wyckoff_canslim_patterns)} Advanced pattern(s)")
                        
                        for pattern in wyckoff_canslim_patterns:
                            signal = pattern.get('signal', 'NEUTRAL')
                            pattern_name = pattern.get('pattern', 'Unknown')
                            
                            # Determine icon and styling
                            if 'VCP' in pattern_name:
                                icon = "🔥"
                                color = "orange"
                            elif 'Darvas' in pattern_name:
                                icon = "📦"
                                color = "blue"
                            elif 'CANSLIM' in pattern_name:
                                icon = "💎"
                                color = "blue"
                            elif 'Accumulation' in pattern_name:
                                icon = "📊"
                                color = "green"
                            elif 'Distribution' in pattern_name:
                                icon = "📉"
                                color = "red"
                            elif 'Inverse Head' in pattern_name:
                                icon = "🔄"
                                color = "green"
                            elif 'Triple Top' in pattern_name:
                                icon = "🔻🔻🔻"
                                color = "red"
                            elif 'Triple Bottom' in pattern_name:
                                icon = "💚💚💚"
                                color = "green"
                            elif 'Order Block' in pattern_name:
                                icon = "🏦"
                                color = "purple"
                            elif 'Elliott Wave' in pattern_name:
                                icon = "🌊"
                                color = "blue"
                            elif 'Mean Reversion' in pattern_name:
                                icon = "📉📈"
                                color = "purple"
                            else:
                                icon = "📈"
                                color = "gray"
                            
                            with st.expander(f"{icon} {pattern_name} - {signal}", expanded=True):
                                st.markdown(f"**Description:** {pattern['description']}")
                                st.markdown(f"**Action:** {pattern['action']}")
                                
                                # Special display for VCP
                                if 'VCP' in pattern_name:
                                    st.info("🔥 **VCP** = Mark Minervini's Volatility Contraction Pattern-VCP")
                                    
                                    if 'contraction_data' in pattern:
                                        pullbacks = pattern['contraction_data'].get('pullbacks', [])
                                        if pullbacks:
                                            st.markdown("**Contraction Analysis:**")
                                            for idx, pb in enumerate(pullbacks, 1):
                                                st.markdown(f"- Base {idx}: **{pb*100:.1f}%** pullback")
                                            
                                            st.success("✅ Volatility contracting - tightening action confirmed!")
                                
                                # Special display for Darvas Box
                                if 'Darvas' in pattern_name:
                                    st.info("📦 **Darvas Box** = Nicolas Darvas's box theory")
                                    
                                    if 'box_data' in pattern:
                                        box = pattern['box_data']
                                        st.markdown(f"""
                                        **Box Parameters:**
                                        - 📦 **Box Top (Ceiling)**: ₹{box['top']:.2f}
                                        - 📦 **Box Bottom (Floor)**: ₹{box['bottom']:.2f}
                                        - 📦 **Box Range**: {((box['top'] - box['bottom']) / box['top'] * 100):.1f}%
                                        - 📦 **Periods Held**: {box['periods_held']} days
                                        """)
                                        
                                        st.success("✅ Box drawn on chart - look for breakout above ceiling!")
                                
                                # Special display for CANSLIM
                                if 'CANSLIM' in pattern_name:
                                    st.info("💎 **CANSLIM** = Growth stock methodology by William O'Neil")
                                    
                                    # Display CANSLIM score
                                    canslim_score = pattern.get('score', 0) * 100
                                    st.progress(pattern.get('score', 0))
                                    st.markdown(f"**CANSLIM Score: {canslim_score:.0f}%** (Need 65%+ to qualify)")
                                
                                # Special display for Wyckoff
                                if 'Wyckoff' in pattern_name:
                                    if 'Accumulation' in pattern_name:
                                        st.success("📊 **Smart Money Accumulation** - Institutions buying quietly")
                                        st.markdown("""
                                        **Wyckoff Phases:**
                                        - ✅ Phase A: Selling Climax completed
                                        - ✅ Phase B: Trading range built
                                        - ✅ Phase C: Spring (final shakeout) done
                                        - ✅ Phase D: Sign of Strength appearing
                                        - 🎯 Phase E: Markup phase ready
                                        """)
                                    else:
                                        st.warning("📉 **Smart Money Distribution** - Institutions selling into strength")
                                        st.markdown("""
                                        **Wyckoff Phases:**
                                        - ✅ Phase A: Buying Climax completed
                                        - ✅ Phase B: Range at top built
                                        - ✅ Phase C: UTAD (upthrust) done
                                        - ✅ Phase D: Sign of Weakness appearing
                                        - 🎯 Phase E: Markdown phase ready
                                        """)
                                
                                # Special display for Inverse Head and Shoulders
                                if 'Inverse Head' in pattern_name:
                                    st.success("🔄 **Inverse H&S** = Bullish reversal (opposite of Head & Shoulders)")
                                    st.markdown("""
                                    **Pattern Structure:**
                                    - Left Shoulder → Deeper Head → Right Shoulder
                                    - Neckline resistance must break
                                    - Volume should increase through pattern
                                    - Strong reversal signal from downtrend
                                    """)
                                
                                # Special display for Triple Patterns
                                if 'Triple Top' in pattern_name:
                                    st.warning("🔻🔻🔻 **Triple Top** = Very strong bearish reversal")
                                    st.markdown("""
                                    **Pattern Strength:**
                                    - 3 failed attempts at resistance
                                    - Higher reliability than Double Top
                                    - Volume declining on each peak
                                    - Major distribution pattern
                                    """)
                                
                                if 'Triple Bottom' in pattern_name:
                                    st.success("💚💚💚 **Triple Bottom** = Very strong bullish reversal")
                                    st.markdown("""
                                    **Pattern Strength:**
                                    - 3 successful tests of support
                                    - Higher reliability than Double Bottom
                                    - Volume increasing shows accumulation
                                    - Major reversal pattern
                                    """)
                                
                                # Special display for Order Blocks
                                if 'Order Block' in pattern_name:
                                    ob_type = pattern.get('order_block_data', {}).get('type', 'UNKNOWN')
                                    if ob_type == 'BULLISH':
                                        st.success("🏦 **Bullish Order Block** = Smart Money / ICT Concept")
                                    else:
                                        st.warning("🏦 **Bearish Order Block** = Smart Money / ICT Concept")
                                    
                                    st.markdown("""
                                    **Order Block Theory:**
                                    - Represents institutional order zones
                                    - Last opposite candle before big move
                                    - Price often returns to test these zones
                                    - ICT (Inner Circle Trader) methodology
                                    - ✅ Zone drawn on chart
                                    """)
                                    
                                    if 'order_block_data' in pattern:
                                        ob = pattern['order_block_data']
                                        st.markdown(f"""
                                        **Block Zone:**
                                        - 🏦 High: ₹{ob['high']:.2f}
                                        - 🏦 Low: ₹{ob['low']:.2f}
                                        - Type: {ob['type']}
                                        """)
                                
                                # Special display for Elliott Wave
                                if 'Elliott Wave' in pattern_name:
                                    st.info("🌊 **Elliott Wave** = Ralph Nelson Elliott's Wave Theory")
                                    st.markdown("""
                                    **Wave Theory Basics:**
                                    - 5-wave impulse in trend direction (1-2-3-4-5)
                                    - 3-wave correction against trend (A-B-C)
                                    - Wave 3 is never the shortest
                                    - Wave 2 never retraces >100% of Wave 1
                                    - Wave 4 doesn't overlap Wave 1
                                    """)
                                    
                                    if 'elliott_data' in pattern:
                                        ew = pattern['elliott_data']
                                        st.markdown(f"**Current Position:** {ew.get('wave_position', 'Unknown')}")
                                        
                                        if 'correction_levels' in ew:
                                            st.markdown("**Fibonacci Correction Levels:**")
                                            levels = ew['correction_levels']
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("38.2% Level", f"₹{levels['38.2%']:.2f}")
                                            with col2:
                                                st.metric("50% Level", f"₹{levels['50%']:.2f}")
                                            with col3:
                                                st.metric("61.8% Level", f"₹{levels['61.8%']:.2f}")
                                            
                                            st.success("✅ Fibonacci levels drawn on chart")
                                
                                # Special display for Mean Reversion
                                if 'Mean Reversion' in pattern_name:
                                    if 'Bullish' in pattern_name:
                                        st.success("📉📈 **Mean Reversion (Bullish)** = Statistical oversold reversion")
                                    else:
                                        st.warning("📈📉 **Mean Reversion (Bearish)** = Statistical overbought reversion")
                                    
                                    st.markdown("""
                                    **Mean Reversion Strategy:**
                                    - Price has deviated significantly from statistical mean
                                    - Bollinger Bands stretch indicates extreme
                                    - RSI confirms oversold/overbought
                                    - Statistical edge: Price tends to revert to mean
                                    - Works best in ranging/sideways markets
                                    """)
                                    
                                    if 'mean_reversion_data' in pattern:
                                        mr = pattern['mean_reversion_data']
                                        
                                        st.markdown("**Statistical Analysis:**")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Std Deviation", f"{abs(mr['std_deviation']):.2f}σ")
                                        with col2:
                                            st.metric("Mean (SMA-20)", f"₹{mr['sma_20']:.2f}")
                                        with col3:
                                            st.metric("RSI", f"{mr['rsi']:.1f}")
                                        with col4:
                                            expected_move = mr.get('expected_gain', mr.get('expected_decline', 0))
                                            st.metric("Expected Move", f"{expected_move:.1f}%")
                                        
                                        st.markdown("**Bollinger Bands:**")
                                        st.markdown(f"- Upper BB (+2σ): ₹{mr['bb_upper']:.2f}")
                                        st.markdown(f"- Middle BB (Mean): ₹{mr['bb_mid']:.2f}")
                                        st.markdown(f"- Lower BB (-2σ): ₹{mr['bb_lower']:.2f}")
                                        
                                        st.success("✅ Bollinger Bands and Mean drawn on chart")
                                
                                # Display entry/exit points
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    if 'entry_point' in pattern:
                                        entry_label = "📍 SHORT Entry" if signal == 'BEARISH' else "📍 Entry"
                                        st.markdown(f"**{entry_label}:** {pattern.get('entry_point', 'N/A')}")
                                with col2:
                                    if 'stop_loss' in pattern:
                                        st.markdown(f"**🛑 Stop:** {pattern.get('stop_loss', 'N/A')}")
                                with col3:
                                    if 'target_1' in pattern:
                                        st.markdown(f"**🎯 T1:** {pattern.get('target_1', 'N/A')}")
                                with col4:
                                    if 'target_2' in pattern:
                                        st.markdown(f"**🎯 T2:** {pattern.get('target_2', 'N/A')}")
                                
                                # Display rules
                                if pattern.get('rules'):
                                    with st.expander("📋 Trading Rules", expanded=False):
                                        for rule in pattern['rules']:
                                            if rule:  # Skip empty rules
                                                st.markdown(f"- {rule}")
                    else:
                        st.info("No Advanced patterns detected in current timeframe")
                
                # Charts
                st.markdown('<div class="sub-header">📊 Technical Analysis Charts</div>', unsafe_allow_html=True)
                
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Price Action & Indicators", "Volume Profile", "Quantitative Analysis"])
                
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
                
                with chart_tab3:
                    st.markdown("### 🔬 Advanced Quantitative Analysis")
                    
                    # Run quantitative analysis
                    with st.spinner('Running advanced quantitative models...'):
                        quant_results = run_full_quantitative_analysis(analyzer.data)
                    
                    # Fractal Analysis Section
                    st.markdown("---")
                    st.markdown("#### 📐 Fractal Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Hurst Exponent Analysis**")
                        hurst = quant_results['fractal']['hurst_exponent']
                        
                        st.metric("Hurst Exponent", f"{hurst['hurst_exponent']:.3f}")
                        st.metric("Market Behavior", hurst['market_behavior'])
                        st.metric("Confidence", hurst['confidence'])
                        
                        # Interpretation
                        if hurst['market_behavior'] == 'PERSISTENT':
                            st.success(f"🔥 **Trending Market** (H = {hurst['hurst_exponent']:.3f} > 0.5)")
                            st.info(f"📊 {hurst['recommendation']}")
                        elif hurst['market_behavior'] == 'ANTI_PERSISTENT':
                            st.warning(f"📉 **Mean Reverting** (H = {hurst['hurst_exponent']:.3f} < 0.5)")
                            st.info(f"📊 {hurst['recommendation']}")
                        else:
                            st.info(f"🎲 **Random Walk** (H ≈ 0.5)")
                            st.info(f"📊 {hurst['recommendation']}")
                        
                        with st.expander("ℹ️ What is Hurst Exponent?"):
                            st.markdown("""
                            **Hurst Exponent (H) measures long-term memory:**
                            - **H = 0.5**: Random walk (no memory)
                            - **H > 0.5**: Trending/Persistent (momentum)
                            - **H < 0.5**: Mean reverting (anti-momentum)
                            
                            **Trading Implications:**
                            - High H → Use trend-following strategies
                            - Low H → Use mean reversion strategies
                            - H ≈ 0.5 → Market is efficient/random
                            """)
                    
                    with col2:
                        st.markdown("**Fractal Dimension Analysis**")
                        fractal_dim = quant_results['fractal']['fractal_dimension']
                        
                        st.metric("Fractal Dimension", f"{fractal_dim['fractal_dimension']:.3f}")
                        st.metric("Market State", fractal_dim['market_state'])
                        st.metric("Confidence", fractal_dim['confidence'])
                        
                        # Interpretation
                        if fractal_dim['market_state'] == 'TRENDING':
                            st.success(f"📈 **Strong Trend** (FD = {fractal_dim['fractal_dimension']:.3f} < 1.5)")
                            st.info(f"📊 {fractal_dim['recommendation']}")
                        elif fractal_dim['market_state'] == 'RANGE_BOUND':
                            st.warning(f"📊 **Choppy Market** (FD = {fractal_dim['fractal_dimension']:.3f} > 1.7)")
                            st.info(f"📊 {fractal_dim['recommendation']}")
                        else:
                            st.info(f"📉 **Mild Trend** (FD ≈ 1.5)")
                            st.info(f"📊 {fractal_dim['recommendation']}")
                        
                        with st.expander("ℹ️ What is Fractal Dimension?"):
                            st.markdown("""
                            **Fractal Dimension (FD) measures complexity:**
                            - **FD = 1.0**: Perfectly smooth (strong trend)
                            - **FD = 1.5**: Random walk
                            - **FD = 2.0**: Highly irregular (choppy)
                            
                            **Trading Implications:**
                            - Low FD → Strong directional bias
                            - Mid FD → Weak trend or random
                            - High FD → Range-bound, use mean reversion
                            """)
                    
                    # Statistical Estimation Section
                    st.markdown("---")
                    st.markdown("#### 📊 Statistical Estimation (MLE & Bayesian)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Maximum Likelihood Estimation**")
                        
                        mle_normal = quant_results['statistical_estimation']['mle_normal']
                        mle_t = quant_results['statistical_estimation']['mle_students_t']
                        model_comp = quant_results['statistical_estimation']['model_comparison']
                        
                        # Display best model
                        if model_comp['best_model_aic'] == 'STUDENTS_T':
                            st.success("✅ **Best Model: Student's t-distribution**")
                            st.markdown(f"**Degrees of Freedom:** {mle_t['degrees_of_freedom']:.2f}")
                            st.markdown(f"**Tail Heaviness:** {mle_t['tail_heaviness']}")
                            st.info("Market shows fat tails - extreme moves are more common than normal distribution predicts")
                        else:
                            st.success("✅ **Best Model: Normal distribution**")
                            st.info("Returns approximately follow normal distribution")
                        
                        st.metric("Annual Return (MLE)", f"{mle_normal['annual_return']:.2f}%")
                        st.metric("Annual Volatility (MLE)", f"{mle_normal['annual_volatility']:.2f}%")
                        st.metric("Sharpe Ratio", f"{mle_normal['sharpe_ratio']:.3f}")
                        
                        with st.expander("ℹ️ What is MLE?"):
                            st.markdown("""
                            **Maximum Likelihood Estimation:**
                            - Finds parameters that maximize probability of observed data
                            - Student's t-distribution better captures fat tails
                            - Used for risk modeling and option pricing
                            """)
                    
                    with col2:
                        st.markdown("**Bayesian Estimation**")
                        
                        bayesian = quant_results['statistical_estimation']['bayesian']
                        
                        st.metric("Posterior Mean Return", f"{bayesian['posterior_mu']:.6f}")
                        st.metric("Posterior Volatility", f"{bayesian['posterior_sigma']:.6f}")
                        st.metric("Prob of Positive Return", f"{bayesian['prob_positive_return']:.1f}%")
                        
                        st.markdown(f"**95% Credible Interval:**")
                        ci = bayesian['credible_interval_95']
                        st.markdown(f"[{ci[0]:.6f}, {ci[1]:.6f}]")
                        
                        st.metric("Annual Return (Bayesian)", f"{bayesian['annual_return_estimate']:.2f}%")
                        st.metric("Annual Volatility (Bayesian)", f"{bayesian['annual_volatility_estimate']:.2f}%")
                        
                        with st.expander("ℹ️ What is Bayesian Estimation?"):
                            st.markdown("""
                            **Bayesian Estimation:**
                            - Combines prior beliefs with observed data
                            - Provides probability distributions, not point estimates
                            - More robust with limited data
                            - Credible intervals show uncertainty
                            """)
                    
                    # Volatility Modelling Section
                    st.markdown("---")
                    st.markdown("#### 📉 Volatility Modelling")
                    
                    vol_results = quant_results['volatility']
                    
                    # GARCH Model
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**GARCH(1,1) Model**")
                        
                        garch = vol_results['garch']
                        
                        if garch['convergence'] == 'SUCCESS':
                            st.success("✅ GARCH Model Converged")
                            
                            st.metric("Omega (ω)", f"{garch['omega']:.6f}")
                            st.metric("Alpha (α)", f"{garch['alpha']:.4f}")
                            st.metric("Beta (β)", f"{garch['beta']:.4f}")
                            st.metric("Persistence (α + β)", f"{garch['persistence']:.4f}")
                            
                            st.markdown(f"**Volatility Forecast:** {garch['forecast_volatility']:.2f}%")
                            st.markdown(f"**Long-run Volatility:** {garch['long_run_volatility']:.2f}%")
                            
                            if garch['persistence'] > 0.98:
                                st.warning("⚠️ High persistence - volatility shocks are long-lasting")
                            elif garch['persistence'] < 0.90:
                                st.info("✅ Low persistence - volatility mean-reverts quickly")
                            
                            with st.expander("ℹ️ What is GARCH?"):
                                st.markdown("""
                                **GARCH (Generalized ARCH):**
                                - Models time-varying volatility
                                - σ²(t) = ω + α·r²(t-1) + β·σ²(t-1)
                                - Captures volatility clustering
                                - Used in risk management and option pricing
                                
                                **Parameters:**
                                - **ω**: Long-run variance level
                                - **α**: Reaction to recent shocks
                                - **β**: Persistence of volatility
                                """)
                        else:
                            st.error(f"❌ GARCH Convergence: {garch['convergence']}")
                    
                    with col2:
                        st.markdown("**Volatility Comparison**")
                        
                        vol_comp = vol_results['comparison'].tail(1)
                        
                        if not vol_comp.empty:
                            st.markdown("**Current Volatility Estimates:**")
                            for col in vol_comp.columns:
                                val = vol_comp[col].iloc[-1]
                                if not pd.isna(val):
                                    st.metric(col.replace('_', ' '), f"{val:.2f}%")
                            
                            with st.expander("ℹ️ Volatility Estimators"):
                                st.markdown("""
                                **Different Volatility Measures:**
                                - **Simple Vol**: Standard deviation of returns
                                - **EWMA**: Exponentially weighted (recent data weighted more)
                                - **Parkinson**: Uses High-Low range (more efficient)
                                - **Garman-Klass**: Uses OHLC (accounts for gaps)
                                - **Yang-Zhang**: Most efficient OHLC estimator
                                """)
                    
                    # Volatility Regimes
                    st.markdown("**Volatility Regime Analysis**")
                    
                    regimes = vol_results['regimes']
                    current_regime = regimes['Regime'].iloc[-1] if not regimes.empty else 'UNKNOWN'
                    current_vol = regimes['Volatility'].iloc[-1] if not regimes.empty else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Regime", current_regime)
                    with col2:
                        st.metric("Current Volatility", f"{current_vol:.2f}%")
                    with col3:
                        regime_changes = regimes['Regime_Change'].sum()
                        st.metric("Regime Changes (Total)", regime_changes)
                    
                    if current_regime == 'HIGH':
                        st.error("⚠️ **HIGH VOLATILITY REGIME** - Increase position sizing caution, widen stops")
                    elif current_regime == 'LOW':
                        st.success("✅ **LOW VOLATILITY REGIME** - Favorable for entries, tighter stops possible")
                    else:
                        st.info("📊 **MEDIUM VOLATILITY REGIME** - Normal market conditions")
                
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
