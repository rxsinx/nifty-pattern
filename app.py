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
from sklearn.cluster import KMeans
import warnings
from pattern_detector import PatternDetector, format_pattern_summary, get_pattern_statistics
from markov_analysis import HiddenMarkovAnalysis, run_hmm_analysis
from mcmc_analysis import run_mcmc_analysis
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
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff0000; font-weight: bold; }
    .neutral { color: #ffa500; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# AI / ADAPTIVE SUPERTREND — K-MEANS CLUSTERING ENGINE
# ============================================================================

def calculate_adaptive_supertrend(df, atr_period=10, n_clusters=3, lookback_clusters=100):
    """
    AI/Adaptive Supertrend using K-Means Clustering.

    Steps:
      1. Compute ATR (atr_period bars)
      2. K-Means cluster recent ATR values into n_clusters volatility regimes
         (Low → tighter multiplier, High → wider multiplier)
      3. Run the standard Supertrend ratcheting algorithm with the adaptive multiplier
      4. Return enriched DataFrame with AI_Supertrend, AI_ST_Direction, AI_Multiplier,
         ATR_Cluster, AI_ST_Regime columns
    """
    df = df.copy()

    # --- 1. ATR -----------------------------------------------------------
    atr_ind = AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period)
    df['ATR'] = atr_ind.average_true_range()
    df['ATR'] = df['ATR'].bfill()

    # --- 2. K-Means on recent ATR window ----------------------------------
    atr_vals = df['ATR'].dropna().values
    window   = min(lookback_clusters, len(atr_vals))
    atr_win  = atr_vals[-window:].reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(atr_win)

    # Sort cluster labels by centroid value (ascending ATR = ascending volatility)
    centroids  = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centroids)                    # low-to-high order
    cluster_map = {old: new for new, old in enumerate(sorted_idx)}

    all_labels     = kmeans.predict(df['ATR'].values.reshape(-1, 1))
    df['ATR_Cluster'] = np.array([cluster_map[l] for l in all_labels])

    # --- 3. Dynamic multiplier per cluster --------------------------------
    # Cluster 0 (lowest vol) → tightest multiplier; higher cluster → wider
    base_mult = {0: 1.5, 1: 2.5, 2: 3.5}
    for c in range(3, n_clusters):
        base_mult[c] = 3.5 + (c - 2) * 0.5
    df['AI_Multiplier'] = df['ATR_Cluster'].map(base_mult)

    # --- 4. Adaptive Supertrend ratcheting --------------------------------
    hl2      = (df['High'] + df['Low']) / 2.0
    upper_b  = hl2 + df['AI_Multiplier'] * df['ATR']
    lower_b  = hl2 - df['AI_Multiplier'] * df['ATR']

    upper_f  = upper_b.copy()
    lower_f  = lower_b.copy()
    ai_st    = pd.Series(np.nan, index=df.index)
    ai_dir   = pd.Series(1,      index=df.index)   # 1=bullish, -1=bearish

    close = df['Close'].values

    for i in range(1, len(df)):
        # Upper band: only tighten when we were below it last bar
        upper_f.iloc[i] = (min(upper_b.iloc[i], upper_f.iloc[i-1])
                           if close[i-1] <= upper_f.iloc[i-1]
                           else upper_b.iloc[i])
        # Lower band: only loosen when we were above it last bar
        lower_f.iloc[i] = (max(lower_b.iloc[i], lower_f.iloc[i-1])
                           if close[i-1] >= lower_f.iloc[i-1]
                           else lower_b.iloc[i])

        # Trend direction
        if   close[i] > upper_f.iloc[i-1]: ai_dir.iloc[i] =  1
        elif close[i] < lower_f.iloc[i-1]: ai_dir.iloc[i] = -1
        else:                               ai_dir.iloc[i] = ai_dir.iloc[i-1]

        ai_st.iloc[i] = lower_f.iloc[i] if ai_dir.iloc[i] == 1 else upper_f.iloc[i]

    df['AI_ST_Upper']     = upper_f
    df['AI_ST_Lower']     = lower_f
    df['AI_Supertrend']   = ai_st
    df['AI_ST_Direction'] = ai_dir

    # --- 5. Human-readable regime label -----------------------------------
    regime_map = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
    for c in range(3, n_clusters):
        regime_map[c] = f'Extreme Vol {c}'
    df['AI_ST_Regime'] = df['ATR_Cluster'].map(regime_map)

    return df


def create_adaptive_supertrend_chart(df_full, atr_period=10, n_clusters=3):
    """
    Build the 4-row AI/Adaptive Supertrend chart.

    Row 1 – Candlestick + AI Supertrend + buy/sell signal markers
    Row 2 – ATR coloured by volatility cluster
    Row 3 – Adaptive multiplier in use (purple fill)
    Row 4 – Volume bars + 20-period MA
    """
    df = calculate_adaptive_supertrend(df_full.tail(200).copy(),
                                        atr_period=atr_period,
                                        n_clusters=n_clusters)

    # Buy / Sell signals (direction flip)
    buy_signals  = df[(df['AI_ST_Direction'] ==  1) & (df['AI_ST_Direction'].shift(1) == -1)]
    sell_signals = df[(df['AI_ST_Direction'] == -1) & (df['AI_ST_Direction'].shift(1) ==  1)]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.18, 0.14, 0.18],
        subplot_titles=(
            '🤖 AI Adaptive Supertrend — K-Means Dynamic Multiplier',
            'ATR by Volatility Regime (Cluster)',
            'Adaptive Multiplier in Use',
            'Volume'
        )
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'],  close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'), row=1, col=1)

    # Split ST line into bull / bear to prevent colour bleeding
    bull_mask = df['AI_ST_Direction'] == 1
    bear_mask = df['AI_ST_Direction'] == -1
    bull_st   = df['AI_Supertrend'].where(bull_mask)
    bear_st   = df['AI_Supertrend'].where(bear_mask)

    fig.add_trace(go.Scatter(x=df.index, y=bull_st, name='AI-ST (Bull)',
                              line=dict(color='#00e676', width=2.5), mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear_st, name='AI-ST (Bear)',
                              line=dict(color='#ff1744', width=2.5), mode='lines'), row=1, col=1)

    # Background shading by regime cluster
    cluster_colours = {0: 'rgba(0,230,118,0.06)', 1: 'rgba(255,214,0,0.06)', 2: 'rgba(255,23,68,0.06)'}
    prev_cluster, seg_start = None, df.index[0]
    for idx, row_d in df.iterrows():
        c = int(row_d['ATR_Cluster'])
        if c != prev_cluster:
            if prev_cluster is not None:
                fig.add_vrect(x0=seg_start, x1=idx,
                              fillcolor=cluster_colours.get(prev_cluster, 'rgba(200,200,200,0.05)'),
                              layer='below', line_width=0, row=1, col=1)
            seg_start    = idx
            prev_cluster = c
    if prev_cluster is not None:
        fig.add_vrect(x0=seg_start, x1=df.index[-1],
                      fillcolor=cluster_colours.get(prev_cluster, 'rgba(200,200,200,0.05)'),
                      layer='below', line_width=0, row=1, col=1)

    # Buy / Sell markers
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Low'] * 0.992,
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=14, color='#00e676',
                        line=dict(color='white', width=1)),
            text=['BUY'] * len(buy_signals),
            textposition='bottom center',
            textfont=dict(color='#00e676', size=9),
            name='Buy Signal'), row=1, col=1)

    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['High'] * 1.008,
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=14, color='#ff1744',
                        line=dict(color='white', width=1)),
            text=['SELL'] * len(sell_signals),
            textposition='top center',
            textfont=dict(color='#ff1744', size=9),
            name='Sell Signal'), row=1, col=1)

    # ── Row 2: ATR coloured by cluster ────────────────────────────────────
    cluster_line_colours = {0: '#00e676', 1: '#ffd600', 2: '#ff1744'}
    regime_labels = {0: 'ATR Low Vol', 1: 'ATR Med Vol', 2: 'ATR High Vol'}
    for c_id in range(n_clusters):
        atr_seg = df['ATR'].where(df['ATR_Cluster'] == c_id)
        fig.add_trace(go.Scatter(
            x=df.index, y=atr_seg,
            name=regime_labels.get(c_id, f'ATR Cluster {c_id}'),
            line=dict(color=cluster_line_colours.get(c_id, '#aaaaaa'), width=2),
            mode='lines'), row=2, col=1)

    # ── Row 3: Adaptive Multiplier ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df['AI_Multiplier'],
        name='AI Multiplier',
        line=dict(color='#7c4dff', width=2),
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(124,77,255,0.15)'), row=3, col=1)

    # ── Row 4: Volume ─────────────────────────────────────────────────────
    vol_colours = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i]
                   else '#ef5350' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                          marker_color=vol_colours, opacity=0.7), row=4, col=1)
    vol_sma = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_sma, name='Vol SMA-20',
                              line=dict(color='orange', width=1.5, dash='dot')), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=1100,
        showlegend=True,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa')
    )
    fig.update_xaxes(showgrid=False, color='#555')
    fig.update_yaxes(showgrid=True, gridcolor='#1e1e2e', color='#999')
    fig.update_yaxes(title_text='Price (₹)',   row=1, col=1)
    fig.update_yaxes(title_text='ATR',         row=2, col=1)
    fig.update_yaxes(title_text='Multiplier',  row=3, col=1)
    fig.update_yaxes(title_text='Volume',      row=4, col=1)

    return fig, df, buy_signals, sell_signals


def get_ai_st_dashboard(df_ai):
    """Extract latest AI Supertrend metrics for the live-signal dashboard."""
    latest = df_ai.iloc[-1]
    prev   = df_ai.iloc[-2]

    direction    = int(latest['AI_ST_Direction'])
    regime       = str(latest['AI_ST_Regime'])
    multiplier   = float(latest['AI_Multiplier'])
    atr_val      = float(latest['ATR'])
    st_level     = float(latest['AI_Supertrend'])
    close        = float(latest['Close'])
    dist_pct     = abs(close - st_level) / close * 100
    is_new       = direction != int(prev['AI_ST_Direction'])

    # Consecutive bars in current trend
    dirs  = df_ai['AI_ST_Direction'].values
    streak = 1
    for i in range(len(dirs) - 2, -1, -1):
        if dirs[i] == direction:
            streak += 1
        else:
            break

    return dict(
        direction_raw=direction,
        direction='BULLISH 🟢' if direction == 1 else 'BEARISH 🔴',
        regime=regime,
        multiplier=multiplier,
        atr=atr_val,
        st_level=st_level,
        close=close,
        dist_pct=dist_pct,
        is_new_signal=is_new,
        streak=streak,
    )


# ============================================================================
# EXISTING IndianEquityAnalyzer CLASS (unchanged)
# ============================================================================

class IndianEquityAnalyzer:
    """Master Trader Grade Analysis for Indian Equity Market"""

    def __init__(self, symbol, period='1y'):
        self.symbol = symbol
        self.period = period
        self.data   = None
        self.ticker = None
        self.pattern_detector = None

    def fetch_data(self):
        try:
            if not self.symbol.endswith('.NS') and not self.symbol.endswith('.BO'):
                ticker_symbol = f"{self.symbol}.NS"
            else:
                ticker_symbol = self.symbol
            self.ticker = yf.Ticker(ticker_symbol)
            self.data   = self.ticker.history(period=self.period)
            if self.data.empty:
                ticker_symbol = f"{self.symbol.replace('.NS','')}.BO"
                self.ticker   = yf.Ticker(ticker_symbol)
                self.data     = self.ticker.history(period=self.period)
            if not self.data.empty:
                self.calculate_indicators()
                self.pattern_detector = PatternDetector(self.data)
                return True
            return False
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return False

    def calculate_indicators(self):
        df = self.data
        df['SMA_20']  = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_21']  = SMAIndicator(df['Close'], window=21).sma_indicator()
        df['SMA_50']  = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['EMA_10']  = EMAIndicator(df['Close'], window=10).ema_indicator()
        df['EMA_20']  = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['EMA_70']  = EMAIndicator(df['Close'], window=70).ema_indicator()
        macd = MACD(df['Close'])
        df['MACD']        = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist']   = macd.macd_diff()
        df['RSI']         = RSIIndicator(df['Close']).rsi()
        bb = BollingerBands(df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Mid']  = bb.bollinger_mavg()
        df['BB_Low']  = bb.bollinger_lband()
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K']  = stoch.stoch()
        df['Stoch_D']  = stoch.stoch_signal()
        df['ATR']      = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['OBV']      = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['VWAP']     = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        self.data = df

    def detect_volume_profile(self):
        df = self.data.tail(200)
        if len(df) < 20:
            return {'poc_price':0,'value_area_high':0,'value_area_low':0,
                    'high_volume_nodes':[],'low_volume_nodes':[],
                    'volume_distribution':np.array([]),'price_bins':np.array([])}
        num_bins = 50
        bins = np.linspace(df['Low'].min(), df['High'].max(), num_bins)
        volume_at_price = []
        for i in range(len(bins)-1):
            mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i+1])
            volume_at_price.append(df[mask]['Volume'].sum())
        volume_at_price = np.array(volume_at_price)
        if len(volume_at_price) == 0 or volume_at_price.sum() == 0:
            return {'poc_price':df['Close'].iloc[-1],'value_area_high':df['High'].max(),
                    'value_area_low':df['Low'].min(),'high_volume_nodes':[],
                    'low_volume_nodes':[],'volume_distribution':volume_at_price,'price_bins':bins}
        poc_idx   = np.argmax(volume_at_price)
        poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2
        total_volume  = volume_at_price.sum()
        target_volume = total_volume * 0.70
        sorted_indices     = np.argsort(volume_at_price)[::-1]
        cumulative_volume  = 0
        value_area_indices = []
        for idx in sorted_indices:
            cumulative_volume += volume_at_price[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        if value_area_indices:
            value_area_high = bins[max(value_area_indices)+1]
            value_area_low  = bins[min(value_area_indices)]
        else:
            value_area_high = df['High'].max()
            value_area_low  = df['Low'].min()
        threshold_high = np.percentile(volume_at_price[volume_at_price>0],80) if np.any(volume_at_price>0) else 0
        threshold_low  = np.percentile(volume_at_price[volume_at_price>0],20) if np.any(volume_at_price>0) else 0
        high_volume_nodes = bins[:-1][volume_at_price > threshold_high]
        low_volume_nodes  = bins[:-1][volume_at_price < threshold_low]
        return {'poc_price':poc_price,'value_area_high':value_area_high,
                'value_area_low':value_area_low,'high_volume_nodes':high_volume_nodes,
                'low_volume_nodes':low_volume_nodes,'volume_distribution':volume_at_price,'price_bins':bins}

    def detect_chart_patterns(self):
        if not self.pattern_detector:
            return []
        try:
            return self.pattern_detector.detect_all_zanger_patterns()
        except Exception as e:
            st.warning(f"Pattern detection error: {e}")
            return []

    def detect_swing_patterns(self):
        if not self.pattern_detector:
            return []
        try:
            return self.pattern_detector.detect_all_swing_patterns()
        except Exception as e:
            st.warning(f"Swing pattern detection error: {e}")
            return []

    def get_trading_signal(self):
        df      = self.data
        current = df.iloc[-1]
        signals = []
        score   = 0
        if current['Close'] > current['SMA_20']:
            signals.append("✅ Price above 20 SMA (Short-term bullish)"); score += 1
        else:
            signals.append("❌ Price below 20 SMA (Short-term bearish)"); score -= 1
        if current['Close'] > current['SMA_50']:
            signals.append("✅ Price above 50 SMA (Medium-term bullish)"); score += 1
        else:
            signals.append("❌ Price below 50 SMA (Medium-term bearish)"); score -= 1
        if current['Close'] > current['SMA_200']:
            signals.append("✅ Price above 200 SMA (Long-term bullish)"); score += 2
        else:
            signals.append("❌ Price below 200 SMA (Long-term bearish)"); score -= 2
        if current['MACD'] > current['MACD_Signal']:
            signals.append("✅ MACD bullish crossover"); score += 1
        else:
            signals.append("❌ MACD bearish crossover"); score -= 1
        if current['RSI'] > 70:
            signals.append("⚠️ RSI Overbought (>70)"); score -= 1
        elif current['RSI'] < 30:
            signals.append("✅ RSI Oversold (<30) - Potential reversal"); score += 1
        else:
            signals.append(f"✅ RSI Neutral ({current['RSI']:.2f})")
        if current['Volume'] > current['Volume_SMA']:
            signals.append("✅ Above average volume (Strong interest)"); score += 1
        else:
            signals.append("⚠️ Below average volume (Weak interest)")
        if score >= 4:   overall = "🟢 STRONG BUY"
        elif score >= 2: overall = "🟢 BUY"
        elif score >= -1:overall = "🟡 HOLD"
        elif score >= -3:overall = "🔴 SELL"
        else:            overall = "🔴 STRONG SELL"
        return overall, signals, score

    def get_risk_management(self):
        df = self.data
        current_price    = df['Close'].iloc[-1]
        atr              = df['ATR'].iloc[-1]
        stop_loss_atr    = current_price - (2 * atr)
        stop_loss_percent= current_price * 0.98
        stop_loss        = max(stop_loss_atr, stop_loss_percent)
        target_1         = current_price * 1.15
        target_2         = current_price * 1.30
        risk_per_share   = current_price - stop_loss
        return {
            'entry_price':    current_price,
            'stop_loss':      stop_loss,
            'target_1':       target_1,
            'target_2':       target_2,
            'risk_per_share': risk_per_share,
            'risk_reward_1':  (target_1 - current_price) / risk_per_share if risk_per_share > 0 else 0,
            'risk_reward_2':  (target_2 - current_price) / risk_per_share if risk_per_share > 0 else 0,
        }

    def get_company_info(self):
        info = {}
        try:
            ti = self.ticker.info
            info = {
                'name':          ti.get('longName','N/A'),
                'sector':        ti.get('sector','N/A'),
                'industry':      ti.get('industry','N/A'),
                'market_cap':    ti.get('marketCap',0),
                'pe_ratio':      ti.get('trailingPE','N/A'),
                'pb_ratio':      ti.get('priceToBook','N/A'),
                'dividend_yield':ti.get('dividendYield',0),
                'eps':           ti.get('trailingEps','N/A'),
                '52w_high':      ti.get('fiftyTwoWeekHigh',0),
                '52w_low':       ti.get('fiftyTwoWeekLow',0),
                'beta':          ti.get('beta','N/A'),
                'description':   ti.get('longBusinessSummary','N/A'),
            }
        except:
            pass
        return info


# ============================================================================
# CHART HELPERS (unchanged from original)
# ============================================================================

def extract_price_from_string(price_str):
    if isinstance(price_str, (int, float)):
        return float(price_str)
    try:
        import re
        numbers = re.findall(r'[\d.]+', str(price_str))
        if numbers:
            return float(numbers[0])
    except:
        pass
    return None


def draw_patterns_on_chart(fig, patterns, df):
    if not patterns:
        return fig
    last_date = df.index[-1]
    for i, pattern in enumerate(patterns):
        entry_price  = extract_price_from_string(pattern.get('entry_point',''))
        stop_loss    = extract_price_from_string(pattern.get('stop_loss',''))
        target_1     = extract_price_from_string(pattern.get('target_1',''))
        target_2     = extract_price_from_string(pattern.get('target_2',''))
        pattern_name = pattern.get('pattern','Unknown')
        signal       = pattern.get('signal','NEUTRAL')
        color = '#00cc66' if signal=='BULLISH' else '#ff4d4d' if signal=='BEARISH' else '#ffa500'

        if 'Darvas Box' in pattern_name and 'box_data' in pattern:
            bd = pattern['box_data']
            fig.add_shape(type="rect",x0=bd['start_date'],x1=bd['end_date'],
                          y0=bd['bottom'],y1=bd['top'],
                          line=dict(color='black',width=1,dash='dash'),
                          fillcolor='rgba(65,105,225,0.1)',row=1,col=1)
            fig.add_hline(y=bd['top'],line_dash="solid",line_color='#4169E1',line_width=1,
                          annotation_text=f"📦 Box Top: ₹{bd['top']:.2f}",annotation_position="right",row=1,col=1)
            fig.add_hline(y=bd['bottom'],line_dash="solid",line_color='#4169E1',line_width=1,
                          annotation_text=f"📦 Box Bottom: ₹{bd['bottom']:.2f}",annotation_position="right",row=1,col=1)

        if 'Order Block' in pattern_name and 'order_block_data' in pattern:
            od = pattern['order_block_data']
            zone_start   = df.index[int(len(df)*0.7)]
            zone_end     = df.index[-1]
            ob_color     = 'rgba(0,255,0,0.15)' if od['type']=='BULLISH' else 'rgba(255,0,0,0.15)'
            border_color = '#00cc66' if od['type']=='BULLISH' else '#ff4d4d'
            fig.add_shape(type="rect",x0=zone_start,x1=zone_end,y0=od['low'],y1=od['high'],
                          line=dict(color=border_color,width=1,dash='dot'),fillcolor=ob_color,row=1,col=1)
            fig.add_annotation(x=zone_end,y=(od['high']+od['low'])/2,text="<b>OB</b>",
                               showarrow=False,font=dict(color=border_color,size=10,family='Arial Black'),
                               bgcolor='rgba(255,255,255,0.9)',bordercolor=border_color,borderwidth=1,
                               xanchor='left',row=1,col=1)

        if 'Elliott Wave' in pattern_name and 'elliott_data' in pattern:
            ew = pattern['elliott_data']
            if 'correction_levels' in ew:
                levels = ew['correction_levels']
                colors_map = {'38.2%':'#FFD700','50%':'#FF8C00','61.8%':'#FF4500'}
                for ln, lp in levels.items():
                    fig.add_hline(y=lp,line_dash="dot",line_color=colors_map.get(ln,'#FFA500'),
                                  line_width=1,annotation_text=f"Wave C {ln}: ₹{lp:.2f}",
                                  annotation_position="left",row=1,col=1)

        if 'Mean Reversion' in pattern_name and 'mean_reversion_data' in pattern:
            mr = pattern['mean_reversion_data']
            fig.add_hline(y=mr['bb_upper'],line_dash="dash",line_color='#FF6B6B',line_width=2,
                          annotation_text=f"Upper BB: ₹{mr['bb_upper']:.2f} (+2σ)",annotation_position="left",row=1,col=1)
            fig.add_hline(y=mr['bb_mid'],line_dash="solid",line_color='#4ECDC4',line_width=3,
                          annotation_text=f"MEAN (SMA-20): ₹{mr['bb_mid']:.2f}",annotation_position="left",row=1,col=1)
            fig.add_hline(y=mr['bb_lower'],line_dash="dash",line_color='#95E1D3',line_width=2,
                          annotation_text=f"Lower BB: ₹{mr['bb_lower']:.2f} (-2σ)",annotation_position="left",row=1,col=1)
            current_price = df['Close'].iloc[-1]
            std_dev = mr['std_deviation']
            fig.add_annotation(x=df.index[-5] if len(df)>5 else df.index[-1],
                               y=current_price*1.02,text=f"<b>{abs(std_dev):.1f}σ from mean</b>",
                               showarrow=True,arrowhead=2,arrowcolor='#FF6B6B' if std_dev>0 else '#95E1D3',
                               ax=0,ay=-40,bgcolor='rgba(255,255,255,0.9)',
                               bordercolor='#FF6B6B' if std_dev>0 else '#95E1D3',borderwidth=2,
                               font=dict(size=12,color='black'),row=1,col=1)

        if entry_price:
            fig.add_hline(y=entry_price,line_dash="dash",line_color=color,line_width=2,
                          annotation_text=f"📍 ENTRY: ₹{entry_price:.2f}",annotation_position="right",row=1,col=1)
        if stop_loss:
            fig.add_hline(y=stop_loss,line_dash="dot",line_color="red",line_width=2,
                          annotation_text=f"🛑 STOP: ₹{stop_loss:.2f}",annotation_position="right",row=1,col=1)
        if target_1:
            fig.add_hline(y=target_1,line_dash="dot",line_color="green",line_width=2,
                          annotation_text=f"🎯 T1: ₹{target_1:.2f}",annotation_position="right",row=1,col=1)
        if target_2:
            fig.add_hline(y=target_2,line_dash="dot",line_color="darkgreen",line_width=2,
                          annotation_text=f"🎯 T2: ₹{target_2:.2f}",annotation_position="right",row=1,col=1)

        if 'Darvas Box' not in pattern_name and 'Order Block' not in pattern_name:
            fig.add_annotation(x=last_date,y=df['High'].max()*(1-0.05*i),
                               text=f"<b>{pattern_name}</b>",showarrow=True,arrowhead=2,
                               arrowsize=1,arrowwidth=2,arrowcolor=color,ax=-50,ay=-30,
                               bgcolor=color,font=dict(color='white',size=12),
                               bordercolor=color,borderwidth=2,borderpad=4,opacity=0.9,
                               row=1,col=1)
    return fig


def create_candlestick_chart(analyzer, patterns=None):
    df = analyzer.data.tail(200)
    fig = make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=0.05,
                        row_heights=[0.5,0.15,0.15,0.2],
                        subplot_titles=('Price Action with Indicators & Patterns','MACD','RSI','Volume Profile'))
    fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
                                  low=df['Low'],close=df['Close'],name='OHLC'),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['SMA_21'],name='SMA 21',line=dict(color='orange',width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['SMA_50'],name='SMA 50',line=dict(color='blue',width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['SMA_200'],name='SMA 200',line=dict(color='red',width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['EMA_10'],name='EMA 10',line=dict(color='green',width=1.5,dash='dash')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['EMA_70'],name='EMA 70',line=dict(color='black',width=1.5,dash='dash')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['BB_High'],name='BB High',line=dict(color='gray',width=1,dash='dot')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['BB_Low'],name='BB Low',line=dict(color='gray',width=1,dash='dot')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['VWAP'],name='VWAP',line=dict(color='purple',width=1,dash='dot')),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD',line=dict(color='blue',width=1)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['MACD_Signal'],name='Signal',line=dict(color='red',width=1)),row=2,col=1)
    colors = ['green' if v>=0 else 'red' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index,y=df['MACD_Hist'],name='MACD Hist',marker_color=colors),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['RSI'],name='RSI',line=dict(color='purple',width=2)),row=3,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color="red",row=3,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color="green",row=3,col=1)
    colors_vol = ['green' if df['Close'].iloc[i]>=df['Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name='Volume',marker_color=colors_vol),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df['Volume_SMA'],name='Vol SMA',line=dict(color='orange',width=2)),row=4,col=1)
    if patterns:
        fig = draw_patterns_on_chart(fig, patterns, df)
    fig.update_layout(title=f'{analyzer.symbol} - Master Trader Analysis with Pattern Detection',
                      xaxis_rangeslider_visible=False,height=1200,showlegend=True,hovermode='x unified')
    fig.update_xaxes(title_text="Date",row=4,col=1)
    fig.update_yaxes(title_text="Price",row=1,col=1)
    fig.update_yaxes(title_text="MACD",row=2,col=1)
    fig.update_yaxes(title_text="RSI",row=3,col=1)
    fig.update_yaxes(title_text="Volume",row=4,col=1)
    return fig


def create_volume_profile_chart(analyzer):
    vp  = analyzer.detect_volume_profile()
    fig = go.Figure()
    if len(vp['volume_distribution']) == 0:
        fig.add_annotation(text="Insufficient data for volume profile",xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False)
        return fig
    price_levels = (vp['price_bins'][:-1]+vp['price_bins'][1:])/2
    fig.add_trace(go.Bar(y=price_levels,x=vp['volume_distribution'],orientation='h',name='Volume at Price',
                          marker=dict(color='lightblue',line=dict(color='blue',width=1))))
    fig.add_hline(y=vp['poc_price'],line_dash="solid",line_color="black",annotation_text="POC",line_width=3)
    fig.add_hrect(y0=vp['value_area_low'],y1=vp['value_area_high'],line_width=0,fillcolor="red",opacity=0.2,
                  annotation_text="Value Area",annotation_position="right")
    for hvn in vp['high_volume_nodes'][:3]:
        fig.add_hline(y=hvn,line_dash="dash",line_color="green",annotation_text="HVN",line_width=1)
    for lvn in vp['low_volume_nodes'][:3]:
        fig.add_hline(y=lvn,line_dash="dash",line_color="orange",annotation_text="LVN",line_width=1)
    fig.update_layout(title='Volume Profile Analysis',xaxis_title='Volume',yaxis_title='Price',height=600,showlegend=True)
    return fig


# ============================================================================
# MCMC VISUALISATION HELPERS
# ============================================================================

def create_mcmc_fan_chart(forecast_summary: Dict, symbol: str) -> go.Figure:
    """
    Build the MCMC price fan chart with nested credible-interval bands
    plus a sample of individual Monte Carlo paths.
    """
    dates   = forecast_summary['forecast_dates']
    fan     = forecast_summary['fan_bands']
    paths   = forecast_summary['sample_paths']   # (≤200, horizon)
    cp      = forecast_summary['current_price']

    # Prepend today's price so the fan starts from now
    import datetime
    today = dates[0] - datetime.timedelta(days=1)
    all_dates = [today] + list(dates)

    def prepend(band_list):
        return [cp] + list(band_list)

    fig = go.Figure()

    # ── 60 individual sample paths (light, thin) ──────────────────────────
    n_show = min(60, paths.shape[0])
    for i in range(n_show):
        path_y = prepend(paths[i])
        fig.add_trace(go.Scatter(
            x=all_dates, y=path_y,
            mode='lines',
            line=dict(color='rgba(150,180,255,0.07)', width=1),
            showlegend=False, hoverinfo='skip'))

    # ── 95 % credible band ────────────────────────────────────────────────
    p2_5  = prepend(fan['2.5'])
    p97_5 = prepend(fan['97.5'])
    fig.add_trace(go.Scatter(
        x=all_dates + all_dates[::-1],
        y=p97_5 + p2_5[::-1],
        fill='toself',
        fillcolor='rgba(100,149,237,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Credible Interval',
        hoverinfo='skip'))

    # ── 80 % credible band ────────────────────────────────────────────────
    p10   = prepend(fan['10'])
    p90   = prepend(fan['90'])
    fig.add_trace(go.Scatter(
        x=all_dates + all_dates[::-1],
        y=p90 + p10[::-1],
        fill='toself',
        fillcolor='rgba(100,149,237,0.22)',
        line=dict(color='rgba(0,0,0,0)'),
        name='80% Credible Interval',
        hoverinfo='skip'))

    # ── 50 % credible band ────────────────────────────────────────────────
    p25 = prepend(fan['25'])
    p75 = prepend(fan['75'])
    fig.add_trace(go.Scatter(
        x=all_dates + all_dates[::-1],
        y=p75 + p25[::-1],
        fill='toself',
        fillcolor='rgba(100,149,237,0.35)',
        line=dict(color='rgba(0,0,0,0)'),
        name='50% Credible Interval',
        hoverinfo='skip'))

    # ── Median path ───────────────────────────────────────────────────────
    p50 = prepend(fan['50'])
    fig.add_trace(go.Scatter(
        x=all_dates, y=p50,
        mode='lines',
        line=dict(color='#00e5ff', width=3),
        name='Median Forecast'))

    # ── Current price anchor ──────────────────────────────────────────────
    fig.add_hline(y=cp, line_dash='dot', line_color='#ffd600', line_width=1,
                  annotation_text=f'Current ₹{cp:.2f}',
                  annotation_position='left')

    # ── Target price ─────────────────────────────────────────────────────
    tp = forecast_summary['target_price']
    fig.add_hline(y=tp, line_dash='dash', line_color='#69ff47', line_width=1.5,
                  annotation_text=f'Median Target ₹{tp:.2f}',
                  annotation_position='right')

    direction_colour = '#69ff47' if forecast_summary['direction'] == 'BULLISH' \
                       else '#ff1744' if forecast_summary['direction'] == 'BEARISH' \
                       else '#ffd600'

    fig.update_layout(
        title=f'⛓️ MCMC Bayesian Price Forecast — {symbol} '
              f'({forecast_summary["forecast_days"]}-Day)',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=550,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='right', x=1),
    )
    fig.update_xaxes(showgrid=False, color='#555')
    fig.update_yaxes(showgrid=True, gridcolor='#1e1e2e', color='#999')
    return fig


def create_posterior_distribution_charts(mcmc_result: Dict, posterior: Dict) -> go.Figure:
    """
    2-panel chart: posterior histogram of μ and σ with
    KDE overlay, MLE reference line, and 95% credible interval shading.
    """
    from plotly.subplots import make_subplots

    mu_samp    = mcmc_result['mu_samples']
    sig_samp   = mcmc_result['sigma_samples']
    mle_mu     = posterior['mle_mu_daily']
    mle_sig    = posterior['mle_sigma_daily']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Posterior P(μ | data)  — Daily Drift',
            'Posterior P(σ | data)  — Daily Volatility'
        )
    )

    # ── μ panel ───────────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=mu_samp,
        nbinsx=80,
        histnorm='probability density',
        marker_color='rgba(100,149,237,0.55)',
        name='μ posterior',
        showlegend=True), row=1, col=1)

    # 95% CI shading via vertical lines
    mu_lo = posterior['mu_ci_95_lo']
    mu_hi = posterior['mu_ci_95_hi']
    mu_m  = posterior['mu_mean']
    fig.add_vline(x=mu_m,   line_color='#00e5ff', line_width=2,
                  annotation_text='Mean', row=1, col=1)
    fig.add_vline(x=mle_mu, line_color='#ffd600', line_width=1.5,
                  line_dash='dot', annotation_text='MLE', row=1, col=1)
    fig.add_vline(x=mu_lo,  line_color='rgba(150,150,150,0.6)',
                  line_width=1, line_dash='dash', row=1, col=1)
    fig.add_vline(x=mu_hi,  line_color='rgba(150,150,150,0.6)',
                  line_width=1, line_dash='dash',
                  annotation_text='95% CI', row=1, col=1)

    # ── σ panel ───────────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=sig_samp,
        nbinsx=80,
        histnorm='probability density',
        marker_color='rgba(255,100,100,0.55)',
        name='σ posterior',
        showlegend=True), row=1, col=2)

    sig_lo = posterior['sigma_ci_95_lo']
    sig_hi = posterior['sigma_ci_95_hi']
    sig_m  = posterior['sigma_mean']
    fig.add_vline(x=sig_m,   line_color='#ff6d00', line_width=2,
                  annotation_text='Mean', row=1, col=2)
    fig.add_vline(x=mle_sig, line_color='#ffd600', line_width=1.5,
                  line_dash='dot', annotation_text='MLE', row=1, col=2)
    fig.add_vline(x=sig_lo,  line_color='rgba(150,150,150,0.6)',
                  line_width=1, line_dash='dash', row=1, col=2)
    fig.add_vline(x=sig_hi,  line_color='rgba(150,150,150,0.6)',
                  line_width=1, line_dash='dash',
                  annotation_text='95% CI', row=1, col=2)

    fig.update_layout(
        title='MCMC Posterior Distributions — Parameter Uncertainty',
        height=380,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, color='#555')
    fig.update_yaxes(showgrid=True, gridcolor='#1e1e2e', color='#999',
                     title_text='Density')
    fig.update_xaxes(title_text='μ (daily drift)', row=1, col=1)
    fig.update_xaxes(title_text='σ (daily volatility)', row=1, col=2)
    return fig


def create_trace_plots(mcmc_result: Dict) -> go.Figure:
    """
    Trace plots for all chains — visual convergence check.
    Good chains look like 'fuzzy caterpillars'.
    """
    mu_chains  = mcmc_result['mu_chains']     # (n_chains, n_samples)
    sig_chains = mcmc_result['sigma_chains']
    n_chains   = mu_chains.shape[0]

    fig = make_subplots(rows=2, cols=1,
                         subplot_titles=('Trace: μ (daily drift)',
                                          'Trace: σ (daily volatility)'),
                         vertical_spacing=0.12)

    chain_colours = ['#00e5ff', '#69ff47', '#ff6d00', '#d500f9',
                     '#ffd600', '#ff1744']

    for c in range(n_chains):
        colour = chain_colours[c % len(chain_colours)]
        x_idx  = list(range(len(mu_chains[c])))

        fig.add_trace(go.Scatter(
            x=x_idx, y=mu_chains[c],
            mode='lines',
            line=dict(color=colour, width=0.8),
            name=f'Chain {c+1} μ',
            legendgroup=f'chain{c}',
            showlegend=True), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_idx, y=sig_chains[c],
            mode='lines',
            line=dict(color=colour, width=0.8),
            name=f'Chain {c+1} σ',
            legendgroup=f'chain{c}',
            showlegend=False), row=2, col=1)

    fig.update_layout(
        title='MCMC Trace Plots — Chain Mixing & Stationarity',
        height=450,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        hovermode='x unified',
    )
    fig.update_xaxes(showgrid=False, color='#555', title_text='Iteration')
    fig.update_yaxes(showgrid=True, gridcolor='#1e1e2e', color='#999')
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="main-header">🎯 Indian Equity Market Pattern analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Master Pattern Analysis</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📊 Analysis Settings")

        symbol = st.text_input(
            "Enter Stock Symbol (NSE)", value="",
            help="Enter NSE symbol without .NS suffix (e.g., RELIANCE, TCS, NIFTY50)"
        )
        period = st.selectbox("Analysis Period",
                              options=['1mo','3mo','6mo','1y','2y','5y'], index=3)
        show_patterns_on_chart = st.checkbox("Show Patterns on Chart", value=True,
                                              help="Draw pattern lines and entry/exit points on the chart")

        st.markdown("---")
        # ── AI Adaptive Supertrend controls ──────────────────────────────
        st.markdown("### 🤖 AI Adaptive Supertrend")
        ai_atr_period = st.slider("ATR Period", min_value=5, max_value=30, value=10, step=1,
                                   help="Period for ATR calculation")
        ai_clusters   = st.slider("Volatility Clusters (K)", min_value=2, max_value=5, value=3, step=1,
                                   help="K-Means clusters:\n2 = simple (Low/High)\n3 = standard (Low/Med/High)\n4-5 = detailed segmentation")
        st.caption("🧠 K-Means auto-selects the optimal ATR multiplier per volatility regime")

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
- ✅ Inv H&S 🔄
- ✅ Triple Top 🔻🔻🔻
- ✅ Triple Bottom 💚💚💚
- ✅ Order Blocks 🏦
- ✅ Elliott Wave 🌊
- ✅ Mean Reversion 📉📈

**TOTAL: 30 Patterns**
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

        analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f'🔄 Analyzing {symbol}...'):
            analyzer = IndianEquityAnalyzer(symbol, period)

            if analyzer.fetch_data():
                # ── Pattern detection ──────────────────────────────────────
                zanger_patterns        = analyzer.detect_chart_patterns()
                swing_patterns         = analyzer.detect_swing_patterns()
                classic_patterns       = analyzer.pattern_detector.detect_all_classic_patterns() if analyzer.pattern_detector else []
                wyckoff_canslim        = analyzer.pattern_detector.detect_all_wyckoff_canslim_patterns() if analyzer.pattern_detector else []
                all_patterns           = zanger_patterns + swing_patterns + classic_patterns + wyckoff_canslim

                # ── AI Adaptive Supertrend (computed once) ─────────────────
                with st.spinner('🤖 Running AI Adaptive Supertrend (K-Means clustering)…'):
                    try:
                        ai_fig, df_ai, ai_buys, ai_sells = create_adaptive_supertrend_chart(
                            analyzer.data, atr_period=ai_atr_period, n_clusters=ai_clusters)
                        ai_dash   = get_ai_st_dashboard(df_ai)
                        ai_st_ok  = True
                    except Exception as e:
                        st.warning(f"AI Supertrend error: {e}")
                        ai_st_ok  = False

                # ── Company Info ───────────────────────────────────────────
                st.markdown('<div class="sub-header">🏢 Company Overview</div>', unsafe_allow_html=True)
                info = analyzer.get_company_info()
                col1,col2,col3,col4 = st.columns(4)
                with col1:
                    st.metric("Company", info.get('name','N/A'))
                    st.metric("Sector",  info.get('sector','N/A'))
                with col2:
                    mc = info.get('market_cap',0)
                    st.metric("Market Cap", f"₹{mc/10000000:.2f} Cr" if mc else 'N/A')
                    st.metric("Industry", info.get('industry','N/A'))
                with col3:
                    st.metric("P/E Ratio", f"{info.get('pe_ratio','N/A'):.2f}" if isinstance(info.get('pe_ratio'),(int,float)) else 'N/A')
                    st.metric("P/B Ratio", f"{info.get('pb_ratio','N/A'):.2f}" if isinstance(info.get('pb_ratio'),(int,float)) else 'N/A')
                with col4:
                    st.metric("Beta", f"{info.get('beta','N/A'):.2f}" if isinstance(info.get('beta'),(int,float)) else 'N/A')
                    dy = info.get('dividend_yield',0)
                    st.metric("Div Yield", f"{dy*100:.2f}%" if dy else 'N/A')

                current = analyzer.data.iloc[-1]
                prev    = analyzer.data.iloc[-2]
                change      = current['Close'] - prev['Close']
                change_pct  = (change / prev['Close']) * 100
                c1,c2,c3,c4,c5 = st.columns(5)
                with c1: st.metric("Current Price", f"₹{current['Close']:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                with c2: st.metric("Day High",       f"₹{current['High']:.2f}")
                with c3: st.metric("Day Low",        f"₹{current['Low']:.2f}")
                with c4: st.metric("52W High",       f"₹{info.get('52w_high',0):.2f}")
                with c5: st.metric("52W Low",        f"₹{info.get('52w_low',0):.2f}")

                # ── AI SUPERTREND LIVE DASHBOARD ──────────────────────────
                if ai_st_ok:
                    st.markdown('<div class="sub-header">🤖 AI Adaptive Supertrend — Live Signal</div>',
                                unsafe_allow_html=True)

                    # Alert banner for new flip
                    if ai_dash['is_new_signal']:
                        if ai_dash['direction_raw'] == 1:
                            st.success("🚀 **NEW BUY SIGNAL** — AI Supertrend just flipped **BULLISH**!")
                        else:
                            st.error("🔻 **NEW SELL SIGNAL** — AI Supertrend just flipped **BEARISH**!")

                    c1,c2,c3,c4,c5 = st.columns(5)
                    with c1: st.metric("AI-ST Direction", "BULLISH" if ai_dash['direction_raw']==1 else "BEARISH",
                                       delta="🟢 Uptrend" if ai_dash['direction_raw']==1 else "🔴 Downtrend")
                    with c2: st.metric("Volatility Regime", ai_dash['regime'])
                    with c3: st.metric("Adaptive Multiplier", f"{ai_dash['multiplier']:.1f}×",
                                       help="Dynamically chosen by K-Means cluster")
                    with c4: st.metric("AI-ST Support/Resist", f"₹{ai_dash['st_level']:.2f}")
                    with c5: st.metric("Distance from Price",   f"{ai_dash['dist_pct']:.2f}%",
                                       delta=f"{ai_dash['streak']} bars in trend")

                    ca,cb,cc = st.columns(3)
                    with ca: st.info(f"📊 **Total AI-ST Buy Signals (last 200 bars):** {len(ai_buys)}")
                    with cb: st.info(f"📊 **Total AI-ST Sell Signals (last 200 bars):** {len(ai_sells)}")
                    with cc:
                        if len(ai_buys)>0 and (len(ai_sells)==0 or ai_buys.index[-1]>ai_sells.index[-1]):
                            last_type = "BUY 🟢"
                        elif len(ai_sells)>0:
                            last_type = "SELL 🔴"
                        else:
                            last_type = "None"
                        st.info(f"📊 **Last Signal:** {last_type}")

                # ── Trading Signal ─────────────────────────────────────────
                st.markdown('<div class="sub-header">🎯 Trading Signal</div>', unsafe_allow_html=True)
                overall, signals, score = analyzer.get_trading_signal()
                col1,col2 = st.columns([1,2])
                with col1:
                    st.markdown(f"### {overall}")
                    st.markdown(f"**Signal Strength: {score}/7**")
                with col2:
                    for sig in signals:
                        st.markdown(sig)

                # ── Risk Management ────────────────────────────────────────
                st.markdown('<div class="sub-header">⚠️ Risk Management Parameters</div>', unsafe_allow_html=True)
                rm = analyzer.get_risk_management()
                col1,col2,col3 = st.columns(3)
                with col1:
                    st.markdown("**Entry & Stop Loss**")
                    st.metric("Entry Price", f"₹{rm['entry_price']:.2f}")
                    st.metric("Stop Loss", f"₹{rm['stop_loss']:.2f}",
                              f"-{((rm['entry_price']-rm['stop_loss'])/rm['entry_price']*100):.2f}%")
                with col2:
                    st.markdown("**Profit Targets**")
                    st.metric("Target 1 (15%)", f"₹{rm['target_1']:.2f}")
                    st.metric("Target 2 (30%)", f"₹{rm['target_2']:.2f}")
                with col3:
                    st.markdown("**Risk/Reward**")
                    st.metric("R:R Target 1", f"1:{rm['risk_reward_1']:.2f}")
                    st.metric("R:R Target 2", f"1:{rm['risk_reward_2']:.2f}")
                    st.metric("Risk/Share",   f"₹{rm['risk_per_share']:.2f}")

                # ── Pattern Tabs ───────────────────────────────────────────
                st.markdown('<div class="sub-header">📈 Chart Pattern Detection</div>', unsafe_allow_html=True)
                tab1,tab2,tab3,tab4 = st.tabs(["Dan Zanger Patterns","Qullamaggie Patterns","Classic Patterns","Advanced Patterns"])

                def render_pattern_list(plist):
                    for pattern in plist:
                        is_bearish = pattern.get('signal')=='BEARISH'
                        icon = "🔻" if is_bearish else "🔹"
                        with st.expander(f"{icon} {pattern['pattern']} - {pattern['signal']}", expanded=True):
                            st.markdown(f"**Description:** {pattern['description']}")
                            st.markdown(f"**Action:** {pattern['action']}")
                            if is_bearish:
                                st.warning("⚠️ **SHORT POSITION** - Profit from declining prices")
                            c1,c2,c3,c4 = st.columns(4)
                            with c1:
                                if 'entry_point' in pattern: st.markdown(f"**📍 Entry:** {pattern['entry_point']}")
                            with c2:
                                if 'stop_loss'   in pattern: st.markdown(f"**🛑 Stop:** {pattern['stop_loss']}")
                            with c3:
                                if 'target_1'    in pattern: st.markdown(f"**🎯 T1:** {pattern['target_1']}")
                            with c4:
                                if 'target_2'    in pattern: st.markdown(f"**🎯 T2:** {pattern['target_2']}")

                with tab1:
                    if zanger_patterns:
                        st.success(f"✅ Found {len(zanger_patterns)} Dan Zanger pattern(s)")
                        render_pattern_list(zanger_patterns)
                    else:
                        st.info("No Dan Zanger patterns detected in current timeframe")
                with tab2:
                    if swing_patterns:
                        st.success(f"✅ Found {len(swing_patterns)} Qullamaggie pattern(s)")
                        render_pattern_list(swing_patterns)
                    else:
                        st.info("No Qullamaggie swing patterns detected in current timeframe")
                with tab3:
                    if classic_patterns:
                        st.success(f"✅ Found {len(classic_patterns)} Classic pattern(s)")
                        bullish_c = [p for p in classic_patterns if p.get('signal')=='BULLISH']
                        bearish_c = [p for p in classic_patterns if p.get('signal')=='BEARISH']
                        neutral_c = [p for p in classic_patterns if p.get('signal') not in ['BULLISH','BEARISH']]
                        if bullish_c:
                            st.markdown("### 🟢 Bullish Patterns")
                            render_pattern_list(bullish_c)
                        if bearish_c:
                            st.markdown("### 🔴 Bearish Patterns (SHORT Opportunities)")
                            render_pattern_list(bearish_c)
                        if neutral_c:
                            st.markdown("### ⚡ Neutral/Breakout Patterns")
                            render_pattern_list(neutral_c)
                    else:
                        st.info("No Classic patterns detected in current timeframe")
                with tab4:
                    if wyckoff_canslim:
                        st.success(f"✅ Found {len(wyckoff_canslim)} Advanced pattern(s)")
                        render_pattern_list(wyckoff_canslim)
                    else:
                        st.info("No Advanced patterns detected in current timeframe")

                # ── Charts ─────────────────────────────────────────────────
                st.markdown('<div class="sub-header">📊 Technical Analysis Charts</div>', unsafe_allow_html=True)

                chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
                    "Price Action & Indicators",
                    "Volume Profile",
                    "🤖 AI Adaptive Supertrend",
                    "⛓️ MCMC Bayesian Forecast",
                    "HMM Price Forecast",
                ])

                with chart_tab1:
                    if show_patterns_on_chart and all_patterns:
                        st.info(f"📌 Showing {len(all_patterns)} detected pattern(s) on chart")
                        fig_candle = create_candlestick_chart(analyzer, all_patterns)
                    else:
                        fig_candle = create_candlestick_chart(analyzer)
                    st.plotly_chart(fig_candle, use_container_width=True)

                with chart_tab2:
                    fig_vp = create_volume_profile_chart(analyzer)
                    st.plotly_chart(fig_vp, use_container_width=True)
                    vp = analyzer.detect_volume_profile()
                    c1,c2,c3 = st.columns(3)
                    with c1: st.metric("Point of Control (POC)", f"₹{vp['poc_price']:.2f}")
                    with c2: st.metric("Value Area High",        f"₹{vp['value_area_high']:.2f}")
                    with c3: st.metric("Value Area Low",         f"₹{vp['value_area_low']:.2f}")

                # ── AI Adaptive Supertrend Chart Tab ───────────────────────
                with chart_tab3:
                    if ai_st_ok:
                        st.plotly_chart(ai_fig, use_container_width=True)

                        # Signal history table
                        st.markdown("### 📋 Recent AI Supertrend Signals")
                        sig_rows = []
                        for idx, row in ai_buys.iterrows():
                            sig_rows.append({'Date': idx.strftime('%Y-%m-%d'), 'Type': '🟢 BUY',
                                             'Price': f"₹{row['Close']:.2f}",
                                             'AI-ST Level': f"₹{row['AI_Supertrend']:.2f}",
                                             'Multiplier': f"{row['AI_Multiplier']:.1f}×",
                                             'Regime': row['AI_ST_Regime']})
                        for idx, row in ai_sells.iterrows():
                            sig_rows.append({'Date': idx.strftime('%Y-%m-%d'), 'Type': '🔴 SELL',
                                             'Price': f"₹{row['Close']:.2f}",
                                             'AI-ST Level': f"₹{row['AI_Supertrend']:.2f}",
                                             'Multiplier': f"{row['AI_Multiplier']:.1f}×",
                                             'Regime': row['AI_ST_Regime']})
                        if sig_rows:
                            sig_df = pd.DataFrame(sig_rows).sort_values('Date',ascending=False).head(15)
                            st.dataframe(sig_df, use_container_width=True, hide_index=True)

                        # Regime distribution
                        st.markdown("### 🎯 Volatility Regime Distribution (last 200 bars)")
                        regime_counts  = df_ai['AI_ST_Regime'].value_counts()
                        regime_emojis  = {'Low Vol':'🟢','Medium Vol':'🟡','High Vol':'🔴'}
                        rcols = st.columns(min(len(regime_counts), 3))
                        for j, (rname, rcount) in enumerate(regime_counts.items()):
                            if j < 3:
                                pct   = rcount / len(df_ai) * 100
                                emoji = regime_emojis.get(rname,'⚫')
                                with rcols[j]:
                                    st.metric(f"{emoji} {rname}", f"{pct:.1f}% of time",
                                              delta=f"{rcount} bars")

                        # Deep explainer
                        with st.expander("ℹ️ How AI/Adaptive Supertrend Works — Full Explanation"):
                            st.markdown(f"""
### 🤖 AI/Adaptive Supertrend — K-Means Clustering Engine

**The core problem with classic Supertrend:**
A fixed multiplier (e.g., always 3.0) is *too tight* in calm markets (causes whipsaws) and *too loose* in volatile markets (gives late signals).

**The AI solution:** Use unsupervised machine learning to detect the current volatility regime and **auto-select** the optimal multiplier in real time.

---

### ⚙️ Algorithm — Step by Step

**Step 1 — ATR Calculation**
Average True Range over **{ai_atr_period} periods** measures raw day-to-day volatility.

**Step 2 — K-Means Clustering (K={ai_clusters})**
The last 100 ATR values are clustered into **{ai_clusters} volatility regimes**:

| Cluster | Regime | Multiplier | Effect |
|---------|--------|-----------|--------|
| 0 | 🟢 Low Vol | **1.5×** | Tight bands — catches early trend changes |
| 1 | 🟡 Medium Vol | **2.5×** | Balanced — normal market conditions |
| 2 | 🔴 High Vol | **3.5×** | Wide bands — filters out whipsaws |

**Step 3 — Dynamic Band Calculation**
```
HL2 = (High + Low) / 2
Upper Band = HL2 + (AI_Multiplier × ATR)
Lower Band = HL2 - (AI_Multiplier × ATR)
```

**Step 4 — Ratcheting (same as classic Supertrend)**
- In an uptrend: the lower band can only **rise** (stop trails up)
- In a downtrend: the upper band can only **fall** (stop trails down)
- Prevents the stop from moving against you

**Step 5 — Direction Flip**
- Close crosses **above** upper band → **BULLISH 🟢**
- Close crosses **below** lower band → **BEARISH 🔴**

---

### 📊 Chart Legend

| Element | Meaning |
|---------|---------|
| 🟢 Green line | AI-ST in bullish mode (support) |
| 🔴 Red line | AI-ST in bearish mode (resistance) |
| 🟢 BG shading | Low volatility regime |
| 🟡 BG shading | Medium volatility regime |
| 🔴 BG shading | High volatility regime |
| ▲ BUY marker | Trend flipped bullish |
| ▼ SELL marker | Trend flipped bearish |
| 🟣 Multiplier panel | Which multiplier is active |
| Green ATR line | ATR in Low Vol cluster |
| Yellow ATR line | ATR in Medium Vol cluster |
| Red ATR line | ATR in High Vol cluster |

---

### ✅ AI vs Fixed Supertrend

| Feature | Fixed (3.0×) | AI Adaptive |
|---------|-------------|-------------|
| Multiplier | Always 3.0 | **Auto 1.5 / 2.5 / 3.5** |
| Low-vol markets | Too wide → late signals | Tight (1.5×) → early |
| High-vol markets | Too tight → whipsaws | Wide (3.5×) → stable |
| Regime awareness | ❌ None | ✅ K-Means |
| Adapts to market | ❌ Static | ✅ Dynamic |

---

### ⚠️ Disclaimer
All indicators are probability tools, not guarantees. Always use proper position sizing and stop losses.
                            """)
                    else:
                        st.error("AI Supertrend could not be computed. Try a longer data period (1y or more).")

                with chart_tab4:
                    # ── MCMC BAYESIAN FORECAST ──────────────────────────────
                    st.markdown("### ⛓️ Markov Chain Monte Carlo (MCMC) Bayesian Price Forecast")

                    # Sidebar-style MCMC controls inside the tab
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        mcmc_days    = st.slider("Forecast Days", 10, 60, 30, 5,
                                                  key='mcmc_days')
                    with mc2:
                        mcmc_chains  = st.slider("MCMC Chains", 2, 4, 4, 1,
                                                  key='mcmc_chains',
                                                  help="More chains = better convergence check")
                    with mc3:
                        mcmc_samples = st.slider("Samples / Chain", 1000, 5000, 3000, 500,
                                                  key='mcmc_samples',
                                                  help="More samples = smoother posterior")

                    with st.spinner('⛓️ Running MCMC sampler — drawing from posterior P(μ,σ | data)...'):
                        try:
                            mcmc_out = run_mcmc_analysis(
                                data          = analyzer.data,
                                forecast_days = mcmc_days,
                                n_samples     = mcmc_samples,
                                n_warmup      = max(500, mcmc_samples // 2),
                                n_chains      = mcmc_chains,
                                n_paths       = 2000,
                                seed          = 42,
                            )
                            mcmc_ok = True
                        except Exception as e:
                            st.error(f"MCMC error: {e}")
                            mcmc_ok = False

                    if mcmc_ok:
                        fs   = mcmc_out['forecast_summary']
                        post = mcmc_out['posterior']
                        risk = mcmc_out['risk_metrics']
                        diag = mcmc_out['diagnostics']
                        mr   = mcmc_out['mcmc_result']

                        # ── Convergence banner ──────────────────────────────
                        if diag['converged']:
                            st.success(
                                f"✅ MCMC Converged — "
                                f"R-hat μ={diag['r_hat_mu']:.4f}, "
                                f"R-hat σ={diag['r_hat_sigma']:.4f} "
                                f"(both < 1.05) | "
                                f"ESS μ={diag['ess_mu']:.0f}, "
                                f"ESS σ={diag['ess_sigma']:.0f} | "
                                f"Accept rate={diag['accept_rate']:.1%}"
                            )
                        else:
                            st.warning(
                                f"⚠️ Convergence uncertain — "
                                f"R-hat μ={diag['r_hat_mu']:.4f}, "
                                f"R-hat σ={diag['r_hat_sigma']:.4f}. "
                                f"Try more samples or chains."
                            )

                        # ── Direction signal ────────────────────────────────
                        direction_col = st.columns(1)[0]
                        if fs['direction'] == 'BULLISH':
                            st.success(
                                f"📈 **BULLISH** — Posterior median target "
                                f"₹{fs['target_price']:.2f} "
                                f"({fs['expected_return']:+.2f}% in {mcmc_days} days) | "
                                f"Prob(profit) = {risk['prob_profit']:.1%}")
                        elif fs['direction'] == 'BEARISH':
                            st.error(
                                f"📉 **BEARISH** — Posterior median target "
                                f"₹{fs['target_price']:.2f} "
                                f"({fs['expected_return']:+.2f}% in {mcmc_days} days) | "
                                f"Prob(loss >5%) = {risk['prob_loss_5pct']:.1%}")
                        else:
                            st.info(
                                f"📊 **NEUTRAL** — Range-bound. "
                                f"95% CI: ₹{fs['ci_95_low']:.2f} – ₹{fs['ci_95_high']:.2f} | "
                                f"Prob(profit) = {risk['prob_profit']:.1%}")

                        # ── Key metrics row ─────────────────────────────────
                        km1, km2, km3, km4, km5, km6 = st.columns(6)
                        with km1:
                            st.metric("Current Price",  f"₹{fs['current_price']:.2f}")
                        with km2:
                            st.metric("Median Target",  f"₹{fs['target_price']:.2f}",
                                      delta=f"{fs['expected_return']:+.2f}%")
                        with km3:
                            st.metric("95% CI Low",     f"₹{fs['ci_95_low']:.2f}")
                        with km4:
                            st.metric("95% CI High",    f"₹{fs['ci_95_high']:.2f}")
                        with km5:
                            st.metric("Ann. Drift",
                                      f"{fs['ann_drift_mean']:+.1f}%",
                                      delta=f"95% CI: {fs['ann_drift_lo']:+.1f}% to {fs['ann_drift_hi']:+.1f}%")
                        with km6:
                            st.metric("Ann. Volatility", f"{fs['ann_volatility']:.1f}%")

                        # ── Fan chart ───────────────────────────────────────
                        st.plotly_chart(
                            create_mcmc_fan_chart(fs, symbol),
                            use_container_width=True)

                        # ── Posterior distributions ─────────────────────────
                        st.markdown("#### 📊 Posterior Parameter Distributions")
                        st.plotly_chart(
                            create_posterior_distribution_charts(mr, post),
                            use_container_width=True)

                        # ── Posterior table ─────────────────────────────────
                        st.markdown("#### 🔢 Posterior Summary Table")
                        post_tbl = pd.DataFrame([
                            {
                                'Parameter': 'μ  (daily drift)',
                                'Posterior Mean':   f"{post['mu_mean']*100:+.4f}%",
                                'Posterior Median': f"{post['mu_median']*100:+.4f}%",
                                'Std Dev':          f"{post['mu_std']*100:.4f}%",
                                '90% CI':           f"[{post['mu_ci_90_lo']*100:+.4f}%, "
                                                    f"{post['mu_ci_90_hi']*100:+.4f}%]",
                                'MLE (point est.)': f"{post['mle_mu_daily']*100:+.4f}%",
                            },
                            {
                                'Parameter': 'σ  (daily volatility)',
                                'Posterior Mean':   f"{post['sigma_mean']*100:.4f}%",
                                'Posterior Median': f"{post['sigma_median']*100:.4f}%",
                                'Std Dev':          f"{post['sigma_std']*100:.4f}%",
                                '90% CI':           f"[{post['sigma_ci_90_lo']*100:.4f}%, "
                                                    f"{post['sigma_ci_90_hi']*100:.4f}%]",
                                'MLE (point est.)': f"{post['mle_sigma_daily']*100:.4f}%",
                            },
                        ])
                        st.dataframe(post_tbl, use_container_width=True, hide_index=True)

                        # ── Risk metrics ────────────────────────────────────
                        st.markdown("#### ⚠️ Bayesian Risk Metrics")
                        rk1, rk2, rk3, rk4 = st.columns(4)
                        with rk1:
                            st.metric("Prob(Profit)",       f"{risk['prob_profit']:.1%}")
                            st.metric("Prob(Gain > 5%)",    f"{risk['prob_gain_5pct']:.1%}")
                        with rk2:
                            st.metric("Prob(Gain > 10%)",   f"{risk['prob_gain_10pct']:.1%}")
                            st.metric("Prob(Loss > 5%)",    f"{risk['prob_loss_5pct']:.1%}")
                        with rk3:
                            var_key  = 'var_95'
                            cvar_key = 'cvar_95'
                            st.metric("95% VaR",
                                      f"{risk.get(var_key, 0)*100:+.2f}%",
                                      help="Worst-case return at 95th percentile of loss")
                            st.metric("95% CVaR (ES)",
                                      f"{risk.get(cvar_key, 0)*100:+.2f}%",
                                      help="Expected shortfall beyond VaR")
                        with rk4:
                            st.metric("50% CI Range",
                                      f"₹{fs['ci_50_low']:.0f}–₹{fs['ci_50_high']:.0f}")
                            st.metric("80% CI Range",
                                      f"₹{fs['ci_80_low']:.0f}–₹{fs['ci_80_high']:.0f}")

                        # ── Trace plots ─────────────────────────────────────
                        st.markdown("#### 🔍 MCMC Trace Plots (Convergence Check)")
                        st.plotly_chart(
                            create_trace_plots(mr),
                            use_container_width=True)
                        st.caption(
                            "✅ Good mixing = chains look like overlapping 'fuzzy caterpillars'. "
                            "If chains separate or trend, increase n_warmup or n_samples.")

                        # ── MCMC explainer ──────────────────────────────────
                        with st.expander("ℹ️ Understanding MCMC Bayesian Forecasting"):
                            st.markdown(f"""
### ⛓️ Markov Chain Monte Carlo (MCMC) — Bayesian Price Forecasting

MCMC is the **gold standard** for Bayesian inference. Instead of fitting a single
"best-fit" parameter estimate, MCMC samples the **full posterior probability
distribution** over parameters, giving you:

> **"Given everything the stock has done historically, what is the complete
> range of plausible drift rates and volatilities — and how likely is each?"**

---

### 🏗️ The Model Architecture

**Return model** — Geometric Brownian Motion (GBM):
```
log(P_t / P_t-1) ~ Normal(μ - 0.5σ², σ²)
```
- μ = instantaneous expected daily return (drift)
- σ = daily volatility (randomness)

**Priors** (what we assume before seeing data):
```
μ ~ Normal(0, 0.10)      — symmetric, centred on zero drift
σ ~ Half-Normal(0, 0.03) — positive only, most vol < 8%/day
```

**Posterior** (updated after seeing {diag['n_obs']} trading days):
```
P(μ, σ | data) ∝ P(data | μ, σ) × P(μ) × P(σ)
                    likelihood          prior
```

---

### ⚙️ The Metropolis-Hastings Sampler

The sampler explores the parameter space like a **smart random walk**:

1. Start at an initial guess (μ₀, σ₀)
2. **Propose** a nearby point (μ*, σ*) using a Gaussian step
3. **Accept** the proposal with probability:
   `min(1, P(μ*, σ*|data) / P(μ, σ|data))`
4. If rejected, stay at current point
5. Repeat {mcmc_samples} × {mcmc_chains} times across {mcmc_chains} chains

The **warmup phase** ({max(500, mcmc_samples // 2)} steps) tunes step sizes
toward the target acceptance rate of 23.4% (optimal for 2-D targets).

---

### 🔬 Convergence Diagnostics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| R-hat μ | {diag['r_hat_mu']:.4f} | < 1.05 | {"✅" if diag['r_hat_mu'] < 1.05 else "⚠️"} |
| R-hat σ | {diag['r_hat_sigma']:.4f} | < 1.05 | {"✅" if diag['r_hat_sigma'] < 1.05 else "⚠️"} |
| ESS μ | {diag['ess_mu']:.0f} | > 400 | {"✅" if diag['ess_mu'] > 400 else "⚠️"} |
| ESS σ | {diag['ess_sigma']:.0f} | > 400 | {"✅" if diag['ess_sigma'] > 400 else "⚠️"} |
| Accept rate | {diag['accept_rate']:.1%} | 20–40% | {"✅" if 0.15 < diag['accept_rate'] < 0.50 else "⚠️"} |

**R-hat (Gelman-Rubin):** Ratio of between-chain to within-chain variance.
Values near 1.0 confirm all chains converged to the same distribution.

**ESS (Effective Sample Size):** Accounts for autocorrelation. ESS = 400
from {diag['n_total_samples']} draws → effective independence of samples.

---

### 📊 Fan Chart Interpretation

| Band | Meaning |
|------|---------|
| 🔵 Dark core (50% CI) | Half of all simulated outcomes fall here |
| 🔵 Middle band (80% CI) | 80% of outcomes — "likely" range |
| 🔵 Outer band (95% CI) | 95% of outcomes — "very likely" range |
| Thin blue lines | 60 individual MC paths drawn from posterior |
| Cyan line | Median (50th percentile) forecast path |

---

### ✅ MCMC vs Other Methods

| Feature | Plain MC | HMM | **MCMC Bayesian** |
|---------|----------|-----|----------------|
| Parameter uncertainty | ❌ Point est. | ❌ | ✅ Full posterior |
| Convergence check | ❌ | ❌ | ✅ R-hat + ESS |
| Credible intervals | ❌ | Approx | ✅ Exact Bayesian |
| Incorporates prior knowledge | ❌ | ❌ | ✅ |
| Regime detection | ❌ | ✅ | ❌ (see HMM tab) |
| Computational cost | Low | Low | Medium |

**Bottom line:** MCMC gives the most statistically rigorous forecast.
The fan chart's width is **data-driven** — it widens when historical data
is noisy and tightens when the drift is persistent.

⚠️ *Past data cannot guarantee future returns. Use for education only.*
                            """)

                with chart_tab5:
                    st.markdown("### 🎲 Hidden Markov Model (HMM) Price Forecast")
                    with st.spinner('Running Hidden Markov Model analysis...'):
                        hmm_results   = run_hmm_analysis(analyzer.data, forecast_days=30)
                    forecast        = hmm_results['forecast']
                    characteristics = hmm_results['characteristics']
                    strategy        = hmm_results['strategy']
                    persistence     = hmm_results['persistence']

                    st.markdown("#### 📊 30-Day Price Forecast")
                    c1,c2,c3,c4 = st.columns(4)
                    with c1:
                        st.metric("Current Price", f"₹{forecast['current_price']:.2f}")
                        st.metric("Target Price",  f"₹{forecast['target_price']:.2f}", f"{forecast['expected_return']:.2f}%")
                    with c2:
                        st.metric("Best Case",   f"₹{forecast['best_case']:.2f}",
                                  f"+{((forecast['best_case']-forecast['current_price'])/forecast['current_price']*100):.2f}%")
                        st.metric("Worst Case",  f"₹{forecast['worst_case']:.2f}",
                                  f"{((forecast['worst_case']-forecast['current_price'])/forecast['current_price']*100):.2f}%")
                    with c3:
                        st.metric("Signal",    strategy['signal'])
                        st.metric("Direction", forecast['direction'])
                    with c4:
                        st.metric("Confidence",         forecast['confidence_level'])
                        st.metric("Expected Volatility", f"{forecast['expected_volatility']:.2f}%")

                    st.markdown("---")
                    st.markdown("#### 🔄 Market Regime Analysis")
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        st.markdown("**Current State**")
                        cs   = forecast['current_state']
                        prob = forecast['current_state_probability']
                        if cs=='BULL':   st.success(f"🟢 {cs} ({prob:.1%})")
                        elif cs=='BEAR': st.error(f"🔴 {cs} ({prob:.1%})")
                        else:            st.info(f"🟡 {cs} ({prob:.1%})")
                        st.markdown(f"**Avg Duration:** {persistence[cs]['avg_duration']:.0f} days")
                    with c2:
                        st.markdown("**Dominant Future Regime**")
                        dom      = forecast['dominant_regime']
                        dom_conf = forecast['regime_confidence']
                        if dom=='BULL':   st.success(f"🟢 {dom} ({dom_conf:.1%})")
                        elif dom=='BEAR': st.error(f"🔴 {dom} ({dom_conf:.1%})")
                        else:             st.info(f"🟡 {dom} ({dom_conf:.1%})")
                        st.markdown(f"- Bull: {forecast['bull_probability']:.1%}")
                        st.markdown(f"- Bear: {forecast['bear_probability']:.1%}")
                        st.markdown(f"- Sideways: {forecast['sideways_probability']:.1%}")
                    with c3:
                        st.markdown("**State Transition Matrix**")
                        tm = forecast['state_transition_matrix']
                        tm_df = pd.DataFrame(tm, columns=['→Bull','→Bear','→Side'],
                                             index=['Bull→','Bear→','Side→'])
                        st.dataframe(tm_df.style.format("{:.1%}"), use_container_width=True)

                    if forecast['direction']=='BULLISH':
                        st.success(f"📈 **BULLISH SIGNAL**: Expected ₹{forecast['target_price']:.2f} in 30 days ({forecast['expected_return']:.2f}% gain)")
                    elif forecast['direction']=='BEARISH':
                        st.error(f"📉 **BEARISH SIGNAL**: Expected ₹{forecast['target_price']:.2f} in 30 days ({forecast['expected_return']:.2f}% loss)")
                    else:
                        st.info(f"📊 **NEUTRAL SIGNAL**: Range-bound ₹{forecast['worst_case']:.2f} – ₹{forecast['best_case']:.2f}")

                # ── Key Metrics ────────────────────────────────────────────
                st.markdown('<div class="sub-header">📋 Key Technical Indicators</div>', unsafe_allow_html=True)
                metrics_df = pd.DataFrame({
                    'Indicator':   ['RSI','MACD','Signal Line','Stochastic %K','Stochastic %D','ATR','OBV'],
                    'Current Value':[
                        f"{current['RSI']:.2f}", f"{current['MACD']:.2f}", f"{current['MACD_Signal']:.2f}",
                        f"{current['Stoch_K']:.2f}", f"{current['Stoch_D']:.2f}",
                        f"{current['ATR']:.2f}", f"{current['OBV']:.0f}"
                    ],
                    'Interpretation':[
                        'Overbought' if current['RSI']>70 else 'Oversold' if current['RSI']<30 else 'Neutral',
                        'Bullish' if current['MACD']>current['MACD_Signal'] else 'Bearish','-',
                        'Overbought' if current['Stoch_K']>80 else 'Oversold' if current['Stoch_K']<20 else 'Neutral',
                        'Overbought' if current['Stoch_D']>80 else 'Oversold' if current['Stoch_D']<20 else 'Neutral',
                        'High Volatility' if current['ATR']>analyzer.data['ATR'].mean()*1.5 else 'Normal',
                        'Accumulation' if current['OBV']>analyzer.data['OBV'].mean() else 'Distribution'
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                # ── Trading Rules ──────────────────────────────────────────
                st.markdown('<div class="sub-header">📖 Master Trader Rules Summary</div>', unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("""
### 🎯 Dan Zanger's Golden Rules
1. **Volume is Everything** — Pattern must break with 3x average volume
2. **8% Absolute Sell Rule** — Cut losses without emotion
3. **Focus on Liquid Leaders** — Trade stocks in strong sectors
4. **Patience Pays** — Wait 7-8 weeks for cup formation
5. **Upper Half Entry** — Handle must be in upper half of cup
6. **Pure Technicals** — Price & volume tell the story
                    """)
                with c2:
                    st.markdown("""
### 🎓 Qullamaggie's Swing Rules
1. **Extreme Discipline** — Rigid adherence prevents emotional mistakes
2. **1% Risk Rule** — Never risk more than 1% of portfolio
3. **Market Leaders Only** — Focus on strongest stocks in strongest groups
4. **ORH Entry** — Opening Range High entry for episodic pivots
5. **VDU = Gold** — Volume Dry Up shows selling exhaustion
6. **Momentum Trading** — Follow institutional money flow
7. **3-5 Day Hold** — Quick profits, trail winners with 10/20 SMA
                    """)

                st.success(f"✅ Analysis completed for {symbol}")

            else:
                st.error(f"❌ Unable to fetch data for {symbol}. Please check the symbol and try again.")
                st.info("💡 Tip: Try adding .NS for NSE stocks or .BO for BSE stocks")


if __name__ == "__main__":
    main()
