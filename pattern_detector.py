"""
Pattern Detection Module for Technical Analysis
================================================

This module contains all chart pattern detection algorithms including:
- Dan Zanger's patterns (Cup and Handle, High Tight Flag, etc.)
- Qullamaggie's swing patterns (Breakout, Episodic Pivot, etc.)
- Classic chart patterns (Head and Shoulders, Triangles, Flags, Wedges, etc.)

Author: Market Analyzer Pro
Version: 3.0 - Added Classic Bearish Patterns with Short Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class PatternDetector:
    """
    Comprehensive pattern detection for technical analysis.
    
    This class implements pattern detection algorithms used by master traders
    including Dan Zanger, Qullamaggie, and classic technical analysis patterns.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the pattern detector with OHLCV data.
        
        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                  and technical indicators
        """
        self.data = data
        
    # ============================================================================
    # DAN ZANGER PATTERNS
    # ============================================================================
    
    def detect_cup_and_handle(self, lookback: int = 100) -> Dict:
        """
        Detect Cup and Handle pattern - Dan Zanger's signature pattern.
        
        Pattern Characteristics:
        - U-shaped cup formation (7-8 weeks minimum)
        - Handle in upper half of cup (1-4 weeks)
        - Volume dry-up in handle
        - Breakout with 3x+ volume
        
        Args:
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary with pattern details and entry/exit points
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 60:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find the cup (minimum price point)
        min_price_idx = np.argmin(prices)
        if min_price_idx < 20 or min_price_idx > len(prices) - 20:
            return {'detected': False, 'score': 0}
        
        # Cup formation analysis
        cup_left = prices[:min_price_idx]
        cup_right = prices[min_price_idx:]
        
        if len(cup_left) < 20 or len(cup_right) < 20:
            return {'detected': False, 'score': 0}
        
        # Cup depth (ideal: 15-40%)
        cup_depth = (max(cup_left[0], cup_right[-1]) - prices[min_price_idx]) / max(cup_left[0], cup_right[-1])
        if 0.15 <= cup_depth <= 0.40:
            score += 0.3
        
        # Handle formation (last 15-25% of data)
        handle_start = int(len(df) * 0.75)
        handle_data = df.iloc[handle_start:]
        
        if len(handle_data) < 10:
            return {'detected': False, 'score': 0}
        
        # Handle should be in upper half of cup
        cup_top = max(cup_left[0], cup_right[-1])
        cup_bottom = prices[min_price_idx]
        cup_mid = (cup_top + cup_bottom) / 2
        
        handle_avg = handle_data['Close'].mean()
        if handle_avg > cup_mid:
            score += 0.2
        
        # Volume dry-up in handle
        cup_volume = df['Volume'].iloc[:handle_start].mean()
        handle_volume = handle_data['Volume'].mean()
        
        if handle_volume < cup_volume * 0.7:
            score += 0.2
        
        # Handle tightness
        handle_range = (handle_data['High'].max() - handle_data['Low'].min()) / handle_avg
        if handle_range < 0.15:
            score += 0.2
        
        # U-shape validation
        left_trend = np.polyfit(range(len(cup_left)), cup_left, 1)[0]
        right_trend = np.polyfit(range(len(cup_right)), cup_right, 1)[0]
        
        if left_trend < 0 and right_trend > 0:
            score += 0.1
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            handle_high = handle_data['High'].max()
            handle_low = handle_data['Low'].min()
            
            return {
                'detected': True,
                'pattern': 'Cup and Handle',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                'score': score,
                'description': 'Most powerful bull market pattern. Handle in upper cup with volume dry-up.',
                'entry_point': f"₹{handle_high * 1.01:.2f} (Breakout above handle)",
                'stop_loss': f"₹{handle_low * 0.98:.2f} (Below handle low)",
                'target_1': f"₹{current_price * 1.15:.2f} (15% gain)",
                'target_2': f"₹{current_price + (cup_top - cup_bottom):.2f} (Cup depth projected)",
                'action': 'BUY on breakout with >3x volume',
                'rules': [
                    'Minimum 7-8 weeks cup formation',
                    'Handle 1-4 weeks',
                    f'Entry: Above ₹{handle_high:.2f}',
                    f'Stop: Below ₹{handle_low:.2f}',
                    'Volume must confirm breakout'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_high_tight_flag(self, lookback: int = 30) -> Dict:
        """
        Detect High Tight Flag pattern - Explosive continuation pattern.
        
        Pattern Characteristics:
        - Strong pole (>20% gain in <4 weeks)
        - Tight flag (<15% of pole height)
        - Volume dry-up during flag
        - Flag above midpoint of pole
        
        Args:
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary with pattern details
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Identify pole (strong uptrend)
        pole_length = min(15, len(df) // 2)
        pole_data = df.head(pole_length)
        pole_gain = (pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0]) / pole_data['Close'].iloc[0]
        
        if pole_gain > 0.20:  # Minimum 20% gain
            score += 0.3
        else:
            return {'detected': False, 'score': 0}
        
        # Flag consolidation
        flag_data = df.tail(len(df) - pole_length)
        flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()
        
        if flag_range < 0.15:  # Tight flag
            score += 0.3
        
        # Volume dry-up
        pole_volume = pole_data['Volume'].mean()
        flag_volume = flag_data['Volume'].mean()
        
        if flag_volume < pole_volume * 0.6:
            score += 0.2
        
        # Flag above pole midpoint
        pole_mid = (pole_data['Close'].iloc[0] + pole_data['Close'].iloc[-1]) / 2
        flag_avg = flag_data['Close'].mean()
        
        if flag_avg > pole_mid:
            score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            pole_high = df['High'].tail(30).max()
            flag_low = flag_data['Low'].min()
            pole_height = pole_high - pole_data['Close'].iloc[0]
            
            return {
                'detected': True,
                'pattern': 'High Tight Flag',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Rare explosive pattern. Pole >20% in <4 weeks.',
                'entry_point': f"₹{pole_high * 1.02:.2f} (Breakout above flag)",
                'stop_loss': f"₹{flag_low * 0.97:.2f} (Below flag)",
                'target_1': f"₹{pole_high + pole_height:.2f} (Pole projected)",
                'target_2': f"₹{pole_high + (pole_height * 1.5):.2f} (1.5x pole)",
                'action': 'BUY with massive volume (>5x average)',
                'rules': [
                    f'Entry: Above ₹{pole_high:.2f}',
                    f'Stop: Below ₹{flag_low:.2f}',
                    'Volume surge required'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_ascending_triangle(self, lookback: int = 30) -> Dict:
        """
        Detect Ascending Triangle - Bullish continuation pattern.
        
        Pattern Characteristics:
        - Flat resistance (horizontal top)
        - Rising support (higher lows)
        - Volume declining during formation
        - Breakout with volume surge
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Resistance line (flat top)
        highs = df['High'].values
        resistance_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        resistance_variance = np.std(highs) / np.mean(highs)
        
        if resistance_variance < 0.02 and abs(resistance_slope) < 0.001:
            score += 0.4
        
        # Support line (rising)
        lows = df['Low'].values
        support_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if support_slope > 0.001:
            score += 0.3
        
        # Volume declining
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.2
        
        # Near breakout
        current_close = df['Close'].iloc[-1]
        resistance_level = np.mean(highs[-5:])
        
        if current_close > resistance_level * 0.98:
            score += 0.1
        
        if score > 0.7:
            resistance = df['High'].max()
            support = df['Low'].min()
            triangle_height = resistance - support
            
            return {
                'detected': True,
                'pattern': 'Ascending Triangle',
                'signal': 'BULLISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Buyers aggressive at resistance. Higher lows = accumulation.',
                'entry_point': f"₹{resistance * 1.02:.2f} (Above resistance)",
                'stop_loss': f"₹{support * 0.98:.2f} (Below support)",
                'target_1': f"₹{resistance + triangle_height:.2f} (Height projected)",
                'target_2': f"₹{resistance + (triangle_height * 1.5):.2f} (1.5x height)",
                'action': 'BUY on resistance breakout with volume',
                'rules': [
                    '2-3 resistance touches',
                    f'Entry: Above ₹{resistance:.2f}',
                    'Volume surge required'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_flat_base(self, lookback: int = 20) -> Dict:
        """
        Detect Flat Base - Institutional accumulation pattern.
        
        Pattern Characteristics:
        - Tight consolidation (<15% range)
        - 5-12 week duration
        - Volume contraction
        - Support holding
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Price range analysis
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].mean()
        
        if price_range < 0.12:
            score += 0.4
        
        # Time analysis
        if 15 <= len(df) <= 60:
            score += 0.2
        
        # Volume contraction
        volume_std = df['Volume'].std()
        volume_mean = df['Volume'].mean()
        
        if volume_std / volume_mean < 0.5:
            score += 0.2
        
        # Support holding
        support_level = df['Low'].min()
        recent_lows = df['Low'].tail(5).values
        
        if all(low > support_level * 0.98 for low in recent_lows):
            score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            base_high = df['High'].max()
            base_low = df['Low'].min()
            
            return {
                'detected': True,
                'pattern': 'Flat Base',
                'signal': 'BULLISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Consolidation = institutional accumulation.',
                'entry_point': f"₹{base_high * 1.02:.2f} (Pivot breakout)",
                'stop_loss': f"₹{base_low * 0.97:.2f} (Below base)",
                'target_1': f"₹{current_price * 1.20:.2f} (20% gain)",
                'target_2': f"₹{current_price * 1.35:.2f} (35% gain)",
                'action': 'BUY on volume-fueled breakout',
                'rules': [
                    '5-12 week consolidation',
                    f'Entry: Above ₹{base_high:.2f}'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_falling_wedge(self, lookback: int = 30) -> Dict:
        """
        Detect Falling Wedge - Bullish reversal pattern.
        
        Pattern Characteristics:
        - Both trendlines declining
        - Range narrowing (converging)
        - Volume declining
        - Upside breakout expected
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        highs = df['High'].values
        lows = df['Low'].values
        
        # Both trendlines declining
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if high_trend < 0 and low_trend < 0:
            score += 0.3
        
        # Converging (range narrowing)
        early_range = df.head(10)['High'].max() - df.head(10)['Low'].min()
        late_range = df.tail(10)['High'].max() - df.tail(10)['Low'].min()
        
        if late_range < early_range * 0.7:
            score += 0.3
        
        # Volume declining
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.2
        
        # Near breakout
        current_range = df['High'].iloc[-1] - df['Low'].iloc[-1]
        if current_range < late_range * 1.1:
            score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            upper_line = df['High'].max()
            lower_line = df['Low'].min()
            wedge_height = upper_line - lower_line
            
            return {
                'detected': True,
                'pattern': 'Falling Wedge',
                'signal': 'BULLISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Bullish reversal. Selling pressure decreasing.',
                'entry_point': f"₹{upper_line * 1.01:.2f} (Break above upper line)",
                'stop_loss': f"₹{lower_line * 0.97:.2f} (Below lower line)",
                'target_1': f"₹{current_price + wedge_height:.2f} (Wedge height up)",
                'target_2': f"₹{current_price + (wedge_height * 1.5):.2f} (1.5x height)",
                'action': 'BUY on upside breakout with volume',
                'rules': [
                    'Both lines declining',
                    f'Entry: Above ₹{upper_line:.2f}'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_double_bottom(self, lookback: int = 40) -> Dict:
        """
        Detect Double Bottom - W-shaped reversal pattern.
        
        Pattern Characteristics:
        - Two bottoms at similar price (within 3%)
        - Peak between bottoms (neckline)
        - Higher volume on first bottom
        - Breakout above neckline
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 40:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find local minima
        minima_indices = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                minima_indices.append(i)
        
        if len(minima_indices) >= 2:
            trough1_idx = minima_indices[-2]
            trough2_idx = minima_indices[-1]
            
            trough1_price = prices[trough1_idx]
            trough2_price = prices[trough2_idx]
            
            price_diff = abs(trough1_price - trough2_price) / trough1_price
            
            if price_diff < 0.03:  # Within 3%
                score += 0.3
            
            if trough2_idx - trough1_idx > 10:
                between_prices = prices[trough1_idx:trough2_idx]
                peak_price = np.max(between_prices)
                
                # Volume analysis
                volume1 = df['Volume'].iloc[trough1_idx]
                volume2 = df['Volume'].iloc[trough2_idx]
                
                if volume1 > volume2:
                    score += 0.2
                
                # Near breakout
                current_price = prices[-1]
                if current_price > peak_price * 0.98:
                    score += 0.3
        
        if score > 0.7:
            bottom = df['Low'].min()
            neckline = df['High'].max()
            pattern_height = neckline - bottom
            
            return {
                'detected': True,
                'pattern': 'Double Bottom',
                'signal': 'BULLISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'W-shaped reversal = strong support.',
                'entry_point': f"₹{neckline * 1.02:.2f} (Neckline breakout)",
                'stop_loss': f"₹{bottom * 0.98:.2f} (Below 2nd bottom)",
                'target_1': f"₹{neckline + pattern_height:.2f} (Height projected)",
                'target_2': f"₹{neckline + (pattern_height * 1.5):.2f} (1.5x height)",
                'action': 'BUY on neckline breakout with volume',
                'rules': [
                    'Two bottoms within 3%',
                    f'Entry: Above ₹{neckline:.2f}'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    # ============================================================================
    # CLASSIC BEARISH PATTERNS (NEW - WITH SHORT SIGNALS)
    # ============================================================================
    
    def detect_head_and_shoulders(self, lookback: int = 60) -> Dict:
        """
        Detect Head and Shoulders - Bearish reversal pattern.
        
        Pattern Characteristics:
        - Left shoulder, higher head, right shoulder
        - Neckline connects the two troughs
        - Volume declining through pattern
        - Breakdown below neckline = SHORT signal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find local maxima (peaks) and minima (troughs)
        peaks = []
        troughs = []
        
        for i in range(5, len(prices) - 5):
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                peaks.append((i, prices[i]))
            if prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6]):
                troughs.append((i, prices[i]))
        
        # Need at least 3 peaks and 2 troughs
        if len(peaks) >= 3 and len(troughs) >= 2:
            # Check last 3 peaks for H&S pattern
            left_shoulder = peaks[-3]
            head = peaks[-2]
            right_shoulder = peaks[-1]
            
            # Head should be higher than both shoulders
            if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                score += 0.3
                
                # Shoulders should be roughly equal (within 5%)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                if shoulder_diff < 0.05:
                    score += 0.2
                
                # Neckline (support level connecting troughs)
                if len(troughs) >= 2:
                    trough1 = troughs[-2]
                    trough2 = troughs[-1]
                    neckline = (trough1[1] + trough2[1]) / 2
                    
                    # Current price near or below neckline
                    current_price = prices[-1]
                    if current_price < neckline * 1.02:
                        score += 0.3
                    
                    # Volume declining
                    if df['Volume'].iloc[-10:].mean() < df['Volume'].iloc[-30:-10].mean():
                        score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            neckline = df['Low'].tail(30).min()
            head_price = df['High'].tail(40).max()
            pattern_height = head_price - neckline
            
            return {
                'detected': True,
                'pattern': 'Head and Shoulders',
                'signal': 'BEARISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Classic bearish reversal. Uptrend exhaustion pattern.',
                'entry_point': f"₹{neckline * 0.98:.2f} (SHORT below neckline)",
                'stop_loss': f"₹{head_price * 1.02:.2f} (Above head)",
                'target_1': f"₹{neckline - pattern_height:.2f} (Height projected down)",
                'target_2': f"₹{neckline - (pattern_height * 1.5):.2f} (1.5x height down)",
                'action': '🔻 SHORT on neckline breakdown with volume',
                'rules': [
                    'Wait for neckline break',
                    f'SHORT Entry: Below ₹{neckline:.2f}',
                    f'Stop: Above ₹{head_price:.2f}',
                    'Volume surge on breakdown'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_double_top(self, lookback: int = 40) -> Dict:
        """
        Detect Double Top - M-shaped bearish reversal pattern.
        
        Pattern Characteristics:
        - Two peaks at similar price (within 3%)
        - Trough between peaks (neckline)
        - Lower volume on second peak
        - Breakdown below neckline = SHORT signal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 40:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find local maxima
        maxima_indices = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                maxima_indices.append(i)
        
        if len(maxima_indices) >= 2:
            peak1_idx = maxima_indices[-2]
            peak2_idx = maxima_indices[-1]
            
            peak1_price = prices[peak1_idx]
            peak2_price = prices[peak2_idx]
            
            price_diff = abs(peak1_price - peak2_price) / peak1_price
            
            if price_diff < 0.03:  # Within 3%
                score += 0.3
            
            if peak2_idx - peak1_idx > 10:
                between_prices = prices[peak1_idx:peak2_idx]
                trough_price = np.min(between_prices)
                
                # Volume analysis (second peak lower volume)
                volume1 = df['Volume'].iloc[peak1_idx]
                volume2 = df['Volume'].iloc[peak2_idx]
                
                if volume2 < volume1:
                    score += 0.2
                
                # Near breakdown
                current_price = prices[-1]
                if current_price < trough_price * 1.02:
                    score += 0.3
                
                # Volume increase on breakdown
                recent_volume = df['Volume'].tail(3).mean()
                avg_volume = df['Volume'].mean()
                if recent_volume > avg_volume * 1.2:
                    score += 0.2
        
        if score > 0.7:
            top = df['High'].max()
            neckline = df['Low'].tail(30).min()
            pattern_height = top - neckline
            
            return {
                'detected': True,
                'pattern': 'Double Top',
                'signal': 'BEARISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'M-shaped bearish reversal = strong resistance.',
                'entry_point': f"₹{neckline * 0.98:.2f} (SHORT below neckline)",
                'stop_loss': f"₹{top * 1.02:.2f} (Above peaks)",
                'target_1': f"₹{neckline - pattern_height:.2f} (Height down)",
                'target_2': f"₹{neckline - (pattern_height * 1.5):.2f} (1.5x height down)",
                'action': '🔻 SHORT on neckline breakdown with volume',
                'rules': [
                    'Two tops within 3%',
                    f'SHORT Entry: Below ₹{neckline:.2f}',
                    f'Stop: Above ₹{top:.2f}'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_descending_triangle(self, lookback: int = 30) -> Dict:
        """
        Detect Descending Triangle - Bearish continuation pattern.
        
        Pattern Characteristics:
        - Flat support (horizontal bottom)
        - Lower highs (descending resistance)
        - Volume declining during formation
        - Breakdown below support = SHORT signal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Support line (flat bottom)
        lows = df['Low'].values
        support_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        support_variance = np.std(lows) / np.mean(lows)
        
        if support_variance < 0.02 and abs(support_slope) < 0.001:
            score += 0.4
        
        # Resistance line (descending)
        highs = df['High'].values
        resistance_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        
        if resistance_slope < -0.001:
            score += 0.3
        
        # Volume declining
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.2
        
        # Near breakdown
        current_close = df['Close'].iloc[-1]
        support_level = np.mean(lows[-5:])
        
        if current_close < support_level * 1.02:
            score += 0.1
        
        if score > 0.7:
            support = df['Low'].min()
            resistance = df['High'].max()
            triangle_height = resistance - support
            
            return {
                'detected': True,
                'pattern': 'Descending Triangle',
                'signal': 'BEARISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Sellers aggressive at support. Lower highs = distribution.',
                'entry_point': f"₹{support * 0.98:.2f} (SHORT below support)",
                'stop_loss': f"₹{resistance * 1.02:.2f} (Above resistance)",
                'target_1': f"₹{support - triangle_height:.2f} (Height down)",
                'target_2': f"₹{support - (triangle_height * 1.5):.2f} (1.5x height down)",
                'action': '🔻 SHORT on support breakdown with volume',
                'rules': [
                    '2-3 support touches',
                    f'SHORT Entry: Below ₹{support:.2f}',
                    'Volume surge required'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_symmetrical_triangle(self, lookback: int = 30) -> Dict:
        """
        Detect Symmetrical Triangle - Neutral pattern (can break either way).
        
        Pattern Characteristics:
        - Converging trendlines (rising support, falling resistance)
        - Volume declining during formation
        - Breakout direction determines signal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        highs = df['High'].values
        lows = df['Low'].values
        
        # Resistance descending
        resistance_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        # Support ascending
        support_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if resistance_slope < -0.0005 and support_slope > 0.0005:
            score += 0.4
        
        # Converging
        early_range = df.head(10)['High'].max() - df.head(10)['Low'].min()
        late_range = df.tail(10)['High'].max() - df.tail(10)['Low'].min()
        
        if late_range < early_range * 0.6:
            score += 0.3
        
        # Volume declining
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.3
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            resistance = df['High'].max()
            support = df['Low'].min()
            triangle_height = resistance - support
            
            return {
                'detected': True,
                'pattern': 'Symmetrical Triangle',
                'signal': 'NEUTRAL',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Coiling pattern. Breakout direction determines trade.',
                'entry_point': f"₹{resistance * 1.02:.2f} (BUY) or ₹{support * 0.98:.2f} (SHORT)",
                'stop_loss': f"Opposite side of triangle",
                'target_1': f"₹{current_price + triangle_height:.2f} (UP) or ₹{current_price - triangle_height:.2f} (DOWN)",
                'target_2': f"₹{current_price + (triangle_height * 1.5):.2f} (UP) or ₹{current_price - (triangle_height * 1.5):.2f} (DOWN)",
                'action': '⚡ WAIT for breakout - BUY above resistance OR SHORT below support',
                'rules': [
                    'Wait for clear breakout',
                    f'BUY: Above ₹{resistance:.2f} OR SHORT: Below ₹{support:.2f}',
                    'Volume surge confirms direction'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_bull_flag(self, lookback: int = 25) -> Dict:
        """
        Detect Bull Flag - Bullish continuation pattern.
        
        Pattern Characteristics:
        - Strong pole (sharp uptrend)
        - Flag (slight downward drift or sideways)
        - Parallel channel forming flag
        - Breakout upward continuation
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Identify pole (strong uptrend)
        pole_length = min(10, len(df) // 2)
        pole_data = df.head(pole_length)
        pole_gain = (pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0]) / pole_data['Close'].iloc[0]
        
        if pole_gain > 0.10:  # Minimum 10% gain
            score += 0.3
        else:
            return {'detected': False, 'score': 0}
        
        # Flag consolidation (slight downward or sideways)
        flag_data = df.tail(len(df) - pole_length)
        flag_slope = np.polyfit(range(len(flag_data)), flag_data['Close'].values, 1)[0]
        
        # Flag should be flat or slightly down (not up too much)
        if flag_slope <= 0.001:
            score += 0.3
        
        # Flag range tight
        flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()
        if flag_range < 0.10:
            score += 0.2
        
        # Volume lower in flag
        pole_volume = pole_data['Volume'].mean()
        flag_volume = flag_data['Volume'].mean()
        
        if flag_volume < pole_volume * 0.7:
            score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            flag_high = flag_data['High'].max()
            flag_low = flag_data['Low'].min()
            pole_height = pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0]
            
            return {
                'detected': True,
                'pattern': 'Bull Flag',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                'score': score,
                'description': 'Bullish continuation. Brief pause before resuming uptrend.',
                'entry_point': f"₹{flag_high * 1.01:.2f} (Breakout above flag)",
                'stop_loss': f"₹{flag_low * 0.98:.2f} (Below flag)",
                'target_1': f"₹{flag_high + pole_height:.2f} (Pole height projected)",
                'target_2': f"₹{flag_high + (pole_height * 1.5):.2f} (1.5x pole)",
                'action': 'BUY on breakout above flag with volume',
                'rules': [
                    f'Entry: Above ₹{flag_high:.2f}',
                    f'Stop: Below ₹{flag_low:.2f}',
                    'Volume surge confirms breakout'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_bear_flag(self, lookback: int = 25) -> Dict:
        """
        Detect Bear Flag - Bearish continuation pattern.
        
        Pattern Characteristics:
        - Strong pole (sharp downtrend)
        - Flag (slight upward drift or sideways)
        - Parallel channel forming flag
        - Breakdown continues downtrend = SHORT signal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Identify pole (strong downtrend)
        pole_length = min(10, len(df) // 2)
        pole_data = df.head(pole_length)
        pole_loss = (pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0]) / pole_data['Close'].iloc[0]
        
        if pole_loss < -0.10:  # Minimum 10% drop
            score += 0.3
        else:
            return {'detected': False, 'score': 0}
        
        # Flag consolidation (slight upward or sideways)
        flag_data = df.tail(len(df) - pole_length)
        flag_slope = np.polyfit(range(len(flag_data)), flag_data['Close'].values, 1)[0]
        
        # Flag should be flat or slightly up (counter-trend bounce)
        if flag_slope >= -0.001:
            score += 0.3
        
        # Flag range tight
        flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()
        if flag_range < 0.10:
            score += 0.2
        
        # Volume lower in flag
        pole_volume = pole_data['Volume'].mean()
        flag_volume = flag_data['Volume'].mean()
        
        if flag_volume < pole_volume * 0.7:
            score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            flag_high = flag_data['High'].max()
            flag_low = flag_data['Low'].min()
            pole_height = abs(pole_data['Close'].iloc[0] - pole_data['Close'].iloc[-1])
            
            return {
                'detected': True,
                'pattern': 'Bear Flag',
                'signal': 'BEARISH',
                'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                'score': score,
                'description': 'Bearish continuation. Brief bounce before resuming downtrend.',
                'entry_point': f"₹{flag_low * 0.99:.2f} (SHORT below flag)",
                'stop_loss': f"₹{flag_high * 1.02:.2f} (Above flag)",
                'target_1': f"₹{flag_low - pole_height:.2f} (Pole height down)",
                'target_2': f"₹{flag_low - (pole_height * 1.5):.2f} (1.5x pole down)",
                'action': '🔻 SHORT on breakdown below flag with volume',
                'rules': [
                    f'SHORT Entry: Below ₹{flag_low:.2f}',
                    f'Stop: Above ₹{flag_high:.2f}',
                    'Volume surge confirms breakdown'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_rising_wedge(self, lookback: int = 30) -> Dict:
        """
        Detect Rising Wedge - Bearish reversal pattern.
        
        Pattern Characteristics:
        - Both trendlines rising
        - Range narrowing (converging)
        - Volume declining
        - Downside breakdown = SHORT signal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        highs = df['High'].values
        lows = df['Low'].values
        
        # Both trendlines rising
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if high_trend > 0 and low_trend > 0:
            score += 0.3
        
        # Converging (range narrowing) - lower line rising faster
        if low_trend > high_trend * 0.7:
            score += 0.3
        
        # Volume declining
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.2
        
        # Near breakdown
        current_price = df['Close'].iloc[-1]
        lower_line = df['Low'].tail(10).min()
        
        if current_price < df['Close'].mean():
            score += 0.2
        
        if score > 0.7:
            upper_line = df['High'].max()
            lower_line = df['Low'].min()
            wedge_height = upper_line - lower_line
            
            return {
                'detected': True,
                'pattern': 'Rising Wedge',
                'signal': 'BEARISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Bearish reversal. Buying pressure weakening despite rising price.',
                'entry_point': f"₹{lower_line * 0.99:.2f} (SHORT below lower line)",
                'stop_loss': f"₹{upper_line * 1.02:.2f} (Above upper line)",
                'target_1': f"₹{lower_line - wedge_height:.2f} (Wedge height down)",
                'target_2': f"₹{lower_line - (wedge_height * 1.5):.2f} (1.5x height down)",
                'action': '🔻 SHORT on downside breakdown with volume',
                'rules': [
                    'Both lines rising but converging',
                    f'SHORT Entry: Below ₹{lower_line:.2f}',
                    f'Stop: Above ₹{upper_line:.2f}'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_pennant(self, lookback: int = 25) -> Dict:
        """
        Detect Pennant - Small symmetrical triangle after strong move.
        
        Pattern Characteristics:
        - Strong directional move (pole)
        - Small converging triangle (pennant)
        - Very short duration (1-3 weeks)
        - Continuation pattern
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Identify pole (strong move)
        pole_length = min(8, len(df) // 3)
        pole_data = df.head(pole_length)
        pole_move = (pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0]) / pole_data['Close'].iloc[0]
        
        is_bullish = pole_move > 0.08  # 8% up
        is_bearish = pole_move < -0.08  # 8% down
        
        if is_bullish or is_bearish:
            score += 0.3
        else:
            return {'detected': False, 'score': 0}
        
        # Pennant formation (small converging triangle)
        pennant_data = df.tail(len(df) - pole_length)
        
        if len(pennant_data) < 8:
            return {'detected': False, 'score': 0}
        
        # Converging pattern
        early_range = pennant_data.head(5)['High'].max() - pennant_data.head(5)['Low'].min()
        late_range = pennant_data.tail(5)['High'].max() - pennant_data.tail(5)['Low'].min()
        
        if late_range < early_range * 0.5:
            score += 0.3
        
        # Small size relative to pole
        pennant_range = (pennant_data['High'].max() - pennant_data['Low'].min())
        pole_range = abs(pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0])
        
        if pennant_range < pole_range * 0.5:
            score += 0.2
        
        # Volume declining in pennant
        pole_volume = pole_data['Volume'].mean()
        pennant_volume = pennant_data['Volume'].mean()
        
        if pennant_volume < pole_volume * 0.7:
            score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            pennant_high = pennant_data['High'].max()
            pennant_low = pennant_data['Low'].min()
            
            if is_bullish:
                return {
                    'detected': True,
                    'pattern': 'Bullish Pennant',
                    'signal': 'BULLISH',
                    'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                    'score': score,
                    'description': 'Bullish continuation. Brief consolidation before resuming uptrend.',
                    'entry_point': f"₹{pennant_high * 1.01:.2f} (Breakout above pennant)",
                    'stop_loss': f"₹{pennant_low * 0.98:.2f} (Below pennant)",
                    'target_1': f"₹{pennant_high + pole_range:.2f} (Pole projected)",
                    'target_2': f"₹{pennant_high + (pole_range * 1.5):.2f} (1.5x pole)",
                    'action': 'BUY on upside breakout with volume',
                    'rules': [
                        f'Entry: Above ₹{pennant_high:.2f}',
                        'Quick move expected',
                        'Volume surge confirms'
                    ]
                }
            else:  # Bearish
                return {
                    'detected': True,
                    'pattern': 'Bearish Pennant',
                    'signal': 'BEARISH',
                    'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                    'score': score,
                    'description': 'Bearish continuation. Brief consolidation before resuming downtrend.',
                    'entry_point': f"₹{pennant_low * 0.99:.2f} (SHORT below pennant)",
                    'stop_loss': f"₹{pennant_high * 1.02:.2f} (Above pennant)",
                    'target_1': f"₹{pennant_low - pole_range:.2f} (Pole down)",
                    'target_2': f"₹{pennant_low - (pole_range * 1.5):.2f} (1.5x pole down)",
                    'action': '🔻 SHORT on downside breakdown with volume',
                    'rules': [
                        f'SHORT Entry: Below ₹{pennant_low:.2f}',
                        'Quick move expected',
                        'Volume surge confirms'
                    ]
                }
        
        return {'detected': False, 'score': score}
    
    # ============================================================================
    # QULLAMAGGIE SWING PATTERNS
    # ============================================================================
    
    def detect_qullamaggie_breakout(self, lookback: int = 20) -> Dict:
        """
        Detect Qullamaggie-style Breakout pattern.
        
        Pattern Characteristics:
        - Stair-step higher lows
        - Volume Dry Up (VDU)
        - Above 10/20 EMA
        - Tight consolidation
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Higher lows (stair-step)
        lows = df['Low'].values
        higher_lows = True
        
        for i in range(1, len(lows)):
            if lows[i] < lows[i-1] * 0.98:
                higher_lows = False
                break
        
        if higher_lows:
            score += 0.3
        
        # Volume Dry Up
        recent_vol = df['Volume'].tail(5).mean()
        avg_vol = df['Volume'].mean()
        
        if recent_vol < avg_vol * 0.6:
            score += 0.3
        
        # Above EMAs (if available)
        if 'EMA_8' in df.columns and 'EMA_21' in df.columns:
            current_price = df['Close'].iloc[-1]
            ema_8 = df['EMA_8'].iloc[-1]
            ema_21 = df['EMA_21'].iloc[-1]
            
            if current_price > ema_8 and current_price > ema_21:
                score += 0.2
        
        # Tight range
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].mean()
        if price_range < 0.15:
            score += 0.2
        
        if score > 0.7:
            return {
                'detected': True,
                'pattern': 'Breakout (High Tight Flag)',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                'score': score,
                'description': 'Stair-step pattern with VDU. "Buyers stepping in early = tightening"',
                'action': 'Enter on ORH (Opening Range High) with volume',
                'rules': [
                    '3-5 days tight consolidation',
                    'Volume Dry Up present',
                    'Above 10/20 EMA',
                    'Institutional quality volume'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_episodic_pivot(self, lookback: int = 10) -> Dict:
        """
        Detect Episodic Pivot (EP) - Gap and Go pattern.
        
        Pattern Characteristics:
        - Gap up >2%
        - Huge volume spike (>3x)
        - Holds above gap all day
        - Follow-through next day
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 10:
            return {'detected': False, 'score': 0}
        
        score = 0
        gap_found = False
        volume_spike_found = False
        
        for i in range(1, len(df)):
            prev_close = df['Close'].iloc[i-1]
            current_open = df['Open'].iloc[i]
            
            gap = (current_open - prev_close) / prev_close
            
            if gap > 0.02:  # 2% gap
                gap_found = True
                
                # Volume spike
                current_volume = df['Volume'].iloc[i]
                avg_prev_volume = df['Volume'].iloc[:i].mean()
                
                if current_volume > avg_prev_volume * 3:
                    volume_spike_found = True
        
        if gap_found:
            score += 0.4
        
        if volume_spike_found:
            score += 0.4
        
        # Follow-through
        if len(df) >= 3:
            if df['Close'].iloc[-1] > df['Open'].iloc[-3]:
                score += 0.2
        
        if score > 0.7:
            return {
                'detected': True,
                'pattern': 'Episodic Pivot (EP)',
                'signal': 'BULLISH',
                'confidence': 'HIGH',
                'score': score,
                'description': 'ORH Entry with huge volume. Gap and Go.',
                'action': 'Enter at ORH, hold for 2-3 day momentum',
                'rules': [
                    'Gap up > 2%',
                    'First 5-min volume > 3x',
                    'Holds above ORH all day',
                    'Follow-through next day'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_parabolic_short(self, lookback: int = 20) -> Dict:
        """
        Detect Parabolic Short - Mean reversion setup.
        
        Pattern Characteristics:
        - >30% move in 2-3 weeks
        - Extended from 10 EMA >15%
        - Volume climax
        - First red day
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Strong recent gains
        price_gain = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        if price_gain > 0.30:
            score += 0.3
        
        # Distance from EMAs
        if 'EMA_8' in df.columns and 'EMA_21' in df.columns:
            current_price = df['Close'].iloc[-1]
            ema_8 = df['EMA_8'].iloc[-1]
            ema_21 = df['EMA_21'].iloc[-1]
            
            deviation_8 = (current_price - ema_8) / ema_8
            deviation_21 = (current_price - ema_21) / ema_21
            
            if deviation_8 > 0.15 or deviation_21 > 0.20:
                score += 0.3
        
        # Volume climax
        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        
        if recent_volume > avg_volume * 2:
            score += 0.2
        
        # First red day
        if len(df) >= 3:
            if df['Close'].iloc[-1] < df['Close'].iloc[-2]:
                score += 0.2
        
        if score > 0.7:
            return {
                'detected': True,
                'pattern': 'Parabolic Short',
                'signal': 'BEARISH',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Vertical move extended from EMAs. "Wait for first crack"',
                'action': '🔻 SHORT on first red day, target EMA reversion',
                'rules': [
                    '>30% move in 2-3 weeks',
                    'Extended from 10 EMA >15%',
                    'Volume climax',
                    'First red day triggers SHORT'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_gap_and_go(self, lookback: int = 5) -> Dict:
        """
        Detect Gap and Go - Earnings/news momentum.
        
        Pattern Characteristics:
        - Gap >5%
        - Volume >5x average
        - Gap holds (no fill)
        - Continuation
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 5:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        if len(df) >= 2:
            gap = (df['Open'].iloc[1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            
            if gap > 0.05:
                score += 0.4
                
                # Volume surge
                gap_volume = df['Volume'].iloc[1]
                prev_volume = df['Volume'].iloc[0]
                
                if gap_volume > prev_volume * 3:
                    score += 0.3
                
                # Gap holds
                day_low = df['Low'].iloc[1]
                prev_close = df['Close'].iloc[0]
                
                if day_low > prev_close:
                    score += 0.3
        
        if score > 0.7:
            return {
                'detected': True,
                'pattern': 'Gap and Go',
                'signal': 'BULLISH',
                'confidence': 'HIGH',
                'score': score,
                'description': 'Earnings/News gap with continuation.',
                'action': 'Enter on gap fill hold or continuation',
                'rules': [
                    'Gap > 5%',
                    'Holds gap all day',
                    'Volume > 5x average',
                    'No gap fill'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_abcd_pattern(self, lookback: int = 40) -> Dict:
        """
        Detect ABCD Harmonic Pattern.
        
        Pattern Characteristics:
        - AB = CD in price and time
        - BC retracement 61.8-78.6% of AB
        - CD extension 127.2-161.8% of BC
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 40:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Swing detection
        swings = []
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                swings.append(('peak', i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                swings.append(('trough', i, prices[i]))
        
        if len(swings) >= 4:
            for i in range(len(swings)-3):
                if (swings[i][0] == 'peak' and swings[i+1][0] == 'trough' and 
                    swings[i+2][0] == 'peak' and swings[i+3][0] == 'trough'):
                    
                    A, B, C, D = swings[i][2], swings[i+1][2], swings[i+2][2], swings[i+3][2]
                    
                    AB, BC, CD = A - B, C - B, C - D
                    
                    if AB > 0 and BC > 0 and CD > 0:
                        bc_ab_ratio = BC / AB
                        cd_bc_ratio = CD / BC
                        
                        if 0.618 <= bc_ab_ratio <= 0.786:
                            score += 0.3
                        if 1.272 <= cd_bc_ratio <= 1.618:
                            score += 0.3
                        
                        ab_cd_ratio = AB / CD
                        if 0.8 <= ab_cd_ratio <= 1.2:
                            score += 0.2
        
        if score > 0.7:
            return {
                'detected': True,
                'pattern': 'ABCD Pattern',
                'signal': 'REVERSAL',
                'confidence': 'MEDIUM',
                'score': score,
                'description': 'Harmonic pattern with Fibonacci ratios.',
                'action': 'Enter at D point completion',
                'rules': [
                    'AB = CD in price/time',
                    'BC retracement 61.8-78.6%',
                    'CD extension 127.2-161.8%',
                    'Volume confirmation'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    # ============================================================================
    # WYCKOFF PATTERNS
    # ============================================================================
    
    def detect_vcp(self, lookback: int = 80) -> Dict:
        """
        Detect VCP (Volatility Contraction Pattern) - Mark Minervini's signature pattern.
        
        Pattern Characteristics:
        - Series of contracting price consolidations (3-4 pullbacks minimum)
        - Each pullback shallower than the previous (declining volatility)
        - Volume dries up during contractions
        - Tight consolidation near highs before breakout
        - Also called "Pocket Pivot" or "Constructive Tightness"
        
        VCP Structure:
        - Base 1: 10-20% pullback
        - Base 2: 5-15% pullback (smaller than Base 1)
        - Base 3: 3-10% pullback (smaller than Base 2)
        - Base 4: 1-5% pullback (tightest)
        - Breakout: Volume expansion
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 60:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        volumes = df['Volume'].values
        
        # Find local peaks (highs) to measure pullbacks
        peaks = []
        troughs = []
        
        for i in range(5, len(prices) - 5):
            if prices[i] == max(prices[i-5:i+6]):
                peaks.append((i, prices[i]))
            if prices[i] == min(prices[i-5:i+6]):
                troughs.append((i, prices[i]))
        
        # Need at least 3 peaks and 3 troughs for VCP
        if len(peaks) >= 3 and len(troughs) >= 3:
            # Analyze last 4 pullbacks
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            # Calculate pullback percentages
            pullbacks = []
            for i in range(len(recent_peaks) - 1):
                if i < len(recent_troughs):
                    peak_price = recent_peaks[i][1]
                    trough_price = recent_troughs[i][1]
                    pullback_pct = (peak_price - trough_price) / peak_price
                    pullbacks.append(pullback_pct)
            
            # Check for contracting volatility (each pullback smaller)
            if len(pullbacks) >= 3:
                contracting = True
                for i in range(len(pullbacks) - 1):
                    if pullbacks[i+1] >= pullbacks[i]:
                        contracting = False
                        break
                
                if contracting:
                    score += 0.4
                    
                    # Verify pullback percentages are in VCP range
                    if pullbacks[0] > 0.08 and pullbacks[-1] < 0.05:  # First >8%, Last <5%
                        score += 0.2
        
        # Check volume contraction during latest base
        last_30 = df.tail(30)
        earlier_30 = df.iloc[-60:-30] if len(df) >= 60 else df.iloc[:-30]
        
        if len(earlier_30) > 0:
            if last_30['Volume'].mean() < earlier_30['Volume'].mean() * 0.7:
                score += 0.2
        
        # Tightness check - last 10 days should be very tight
        last_10 = df.tail(10)
        tight_range = (last_10['High'].max() - last_10['Low'].min()) / last_10['Close'].mean()
        
        if tight_range < 0.05:  # Very tight (<5%)
            score += 0.2
        
        # Price near highs
        current_price = prices[-1]
        recent_high = df['High'].tail(60).max()
        
        if current_price > recent_high * 0.95:
            score += 0.1
        
        if score > 0.7:
            pivot_point = df['High'].tail(20).max()
            tight_low = last_10['Low'].min()
            base_low = df['Low'].tail(40).min()
            
            # Calculate VCP contraction data for display
            contraction_data = {
                'pullbacks': pullbacks if len(pullbacks) >= 3 else [],
                'bases_count': len(pullbacks)
            }
            
            return {
                'detected': True,
                'pattern': 'VCP (Volatility Contraction Pattern)',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': "Mark Minervini's VCP. Contracting volatility + tightening action.",
                'entry_point': f"₹{pivot_point * 1.01:.2f} (Breakout above pivot)",
                'stop_loss': f"₹{base_low * 0.98:.2f} (Below base low)",
                'target_1': f"₹{current_price * 1.30:.2f} (30% gain - Stage 2 target)",
                'target_2': f"₹{current_price * 1.50:.2f} (50% gain - extended target)",
                'action': 'BUY on tight breakout with volume surge (2x average)',
                'contraction_data': contraction_data,  # For chart visualization
                'rules': [
                    f'VCP Bases Detected: {len(pullbacks)}',
                    f'Base 1: {pullbacks[0]*100:.1f}% pullback' if len(pullbacks) > 0 else '',
                    f'Base 2: {pullbacks[1]*100:.1f}% pullback (smaller)' if len(pullbacks) > 1 else '',
                    f'Base 3: {pullbacks[2]*100:.1f}% pullback (tighter)' if len(pullbacks) > 2 else '',
                    f'Base 4: {pullbacks[3]*100:.1f}% pullback (tightest)' if len(pullbacks) > 3 else '',
                    'Volume dry-up confirmed',
                    f'Entry: Above ₹{pivot_point:.2f} with 2x volume',
                    'Minervini\'s SEPA criteria: Stage 2 uptrend'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_darvas_box(self, lookback: int = 50) -> Dict:
        """
        Detect Darvas Box Pattern - Nicolas Darvas's box theory.
        
        Pattern Characteristics:
        - Price makes new high, then consolidates
        - Top of box = highest high in consolidation
        - Bottom of box = lowest low in consolidation
        - Box must hold for at least 3-5 periods
        - Breakout above box on volume = BUY signal
        - Stop loss just below box bottom
        
        Box Rules:
        - New high must not be violated for 3+ days (ceiling)
        - New low must not be violated for 3+ days (floor)
        - Price oscillates between ceiling and floor
        - Volume contracts during box formation
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        
        # Find the most recent significant high
        highs = df['High'].values
        lows = df['Low'].values
        
        # Look for a consolidation box in last 20 periods
        box_period = 20
        recent_data = df.tail(box_period)
        
        # Find box boundaries
        box_top = recent_data['High'].max()
        box_bottom = recent_data['Low'].min()
        box_range = (box_top - box_bottom) / recent_data['Close'].mean()
        
        # Box should be relatively tight (5-15% range)
        if 0.05 <= box_range <= 0.20:
            score += 0.3
        
        # Check if box top was established early and held
        box_top_idx = recent_data['High'].idxmax()
        periods_since_top = len(recent_data) - recent_data.index.get_loc(box_top_idx)
        
        if periods_since_top >= 3:  # Top held for 3+ periods
            score += 0.2
            
            # Verify no new highs after box top
            after_top = recent_data.loc[box_top_idx:]
            if len(after_top) > 1:
                if after_top['High'].iloc[1:].max() < box_top * 1.001:  # No breakout yet
                    score += 0.1
        
        # Check if box bottom was established and held
        box_bottom_idx = recent_data['Low'].idxmin()
        periods_since_bottom = len(recent_data) - recent_data.index.get_loc(box_bottom_idx)
        
        if periods_since_bottom >= 3:  # Bottom held for 3+ periods
            score += 0.2
        
        # Volume contraction during box
        box_volume = recent_data['Volume'].mean()
        earlier_volume = df.iloc[-40:-20]['Volume'].mean() if len(df) >= 40 else box_volume
        
        if box_volume < earlier_volume * 0.8:
            score += 0.1
        
        # Current price position
        current_price = df['Close'].iloc[-1]
        
        # Bonus: Price near top of box (ready for breakout)
        if current_price > box_top * 0.97:
            score += 0.1
        
        if score > 0.7:
            # Calculate box data for visualization
            box_data = {
                'top': box_top,
                'bottom': box_bottom,
                'start_date': recent_data.index[0],
                'end_date': recent_data.index[-1],
                'periods_held': periods_since_top
            }
            
            return {
                'detected': True,
                'pattern': 'Darvas Box',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': "Nicolas Darvas's box theory. Price confined between ceiling and floor.",
                'entry_point': f"₹{box_top * 1.01:.2f} (Breakout above box top)",
                'stop_loss': f"₹{box_bottom * 0.99:.2f} (Just below box bottom)",
                'target_1': f"₹{box_top + (box_top - box_bottom):.2f} (Box height projected)",
                'target_2': f"₹{box_top + (box_top - box_bottom) * 2:.2f} (2x box height)",
                'action': 'BUY on breakout above box with strong volume',
                'box_data': box_data,  # For chart visualization
                'rules': [
                    f'Box Top (Ceiling): ₹{box_top:.2f}',
                    f'Box Bottom (Floor): ₹{box_bottom:.2f}',
                    f'Box Range: {box_range*100:.1f}%',
                    f'Box held for {periods_since_top} periods',
                    f'Entry: Above ₹{box_top:.2f} with volume',
                    f'Stop: Below ₹{box_bottom:.2f}',
                    'Darvas: Buy new highs in strong stocks',
                    'Never average down, only pyramid up'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_wyckoff_accumulation(self, lookback: int = 60) -> Dict:
        """
        Detect Wyckoff Accumulation Pattern - Smart Money buying.
        
        Pattern Phases:
        - Phase A: Selling Climax (SC) + Automatic Rally (AR)
        - Phase B: Building a Cause (trading range)
        - Phase C: Spring (final shakeout) or Test
        - Phase D: Sign of Strength (SOS) + Last Point of Support (LPS)
        - Phase E: Markup (breakout)
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        volumes = df['Volume'].values
        
        # Phase A: Selling Climax detection
        volume_spike_idx = np.argmax(volumes)
        if volume_spike_idx < len(df) * 0.3:  # Early in period
            volume_spike = volumes[volume_spike_idx]
            avg_volume = np.mean(volumes)
            
            if volume_spike > avg_volume * 2.5:  # High volume selling
                score += 0.2
                
                # Check for price drop before spike
                if volume_spike_idx > 5:
                    price_before = prices[volume_spike_idx - 5]
                    price_at = prices[volume_spike_idx]
                    if price_at < price_before * 0.95:
                        score += 0.1
        
        # Phase B: Trading Range (consolidation)
        middle_section = df.iloc[int(len(df) * 0.3):int(len(df) * 0.7)]
        if len(middle_section) > 10:
            range_pct = (middle_section['High'].max() - middle_section['Low'].min()) / middle_section['Close'].mean()
            
            if range_pct < 0.15:  # Tight range
                score += 0.2
            
            # Volume should decline in Phase B
            if middle_section['Volume'].mean() < df['Volume'].iloc[:int(len(df) * 0.3)].mean() * 0.7:
                score += 0.1
        
        # Phase C: Spring or Test (lower low on lower volume)
        recent_third = df.tail(int(len(df) * 0.3))
        if len(recent_third) > 5:
            recent_low = recent_third['Low'].min()
            middle_low = middle_section['Low'].min() if len(middle_section) > 0 else recent_low
            
            if recent_low < middle_low * 0.99:  # Lower low
                low_idx = recent_third['Low'].idxmin()
                low_volume = df.loc[low_idx, 'Volume']
                
                if low_volume < df['Volume'].mean() * 0.8:  # Lower volume
                    score += 0.2
        
        # Phase D: Sign of Strength
        last_10 = df.tail(10)
        if len(last_10) > 5:
            # Higher highs on increasing volume
            if last_10['Close'].iloc[-1] > last_10['Close'].iloc[0]:
                if last_10['Volume'].tail(5).mean() > last_10['Volume'].head(5).mean():
                    score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            range_low = df['Low'].tail(40).min()
            range_high = df['High'].tail(40).max()
            
            return {
                'detected': True,
                'pattern': 'Wyckoff Accumulation',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Smart Money accumulation. Selling climax → Range → Spring → Markup.',
                'entry_point': f"₹{range_high * 1.01:.2f} (Breakout above range)",
                'stop_loss': f"₹{range_low * 0.98:.2f} (Below spring low)",
                'target_1': f"₹{current_price * 1.20:.2f} (20% gain)",
                'target_2': f"₹{range_high + (range_high - range_low):.2f} (Range projected)",
                'action': 'BUY on Phase E markup with volume',
                'rules': [
                    'Phase A: Selling Climax detected',
                    'Phase B: Trading range built',
                    'Phase C: Spring completed',
                    'Phase D: Sign of Strength present',
                    f'Entry: Above ₹{range_high:.2f}',
                    'Volume expansion required'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_wyckoff_distribution(self, lookback: int = 60) -> Dict:
        """
        Detect Wyckoff Distribution Pattern - Smart Money selling.
        
        Pattern Phases:
        - Phase A: Buying Climax (BC) + Automatic Reaction (AR)
        - Phase B: Building a Cause (trading range at top)
        - Phase C: Upthrust (UTAD) or Test
        - Phase D: Sign of Weakness (SOW) + Last Point of Supply (LPSY)
        - Phase E: Markdown (breakdown)
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        volumes = df['Volume'].values
        
        # Phase A: Buying Climax (high volume at top)
        volume_spike_idx = np.argmax(volumes)
        if volume_spike_idx < len(df) * 0.3:  # Early in period
            volume_spike = volumes[volume_spike_idx]
            avg_volume = np.mean(volumes)
            
            if volume_spike > avg_volume * 2.5:
                score += 0.2
                
                # Check for price spike upward
                if volume_spike_idx > 5:
                    price_before = prices[volume_spike_idx - 5]
                    price_at = prices[volume_spike_idx]
                    if price_at > price_before * 1.05:
                        score += 0.1
        
        # Phase B: Trading Range at top
        middle_section = df.iloc[int(len(df) * 0.3):int(len(df) * 0.7)]
        if len(middle_section) > 10:
            range_pct = (middle_section['High'].max() - middle_section['Low'].min()) / middle_section['Close'].mean()
            
            if range_pct < 0.15:
                score += 0.2
        
        # Phase C: Upthrust (higher high on lower volume)
        recent_third = df.tail(int(len(df) * 0.3))
        if len(recent_third) > 5:
            recent_high = recent_third['High'].max()
            middle_high = middle_section['High'].max() if len(middle_section) > 0 else recent_high
            
            if recent_high > middle_high * 1.01:  # Higher high
                high_idx = recent_third['High'].idxmax()
                high_volume = df.loc[high_idx, 'Volume']
                
                if high_volume < df['Volume'].mean():  # Lower volume
                    score += 0.2
        
        # Phase D: Sign of Weakness
        last_10 = df.tail(10)
        if len(last_10) > 5:
            # Lower lows on increasing volume
            if last_10['Close'].iloc[-1] < last_10['Close'].iloc[0]:
                if last_10['Volume'].tail(5).mean() > last_10['Volume'].head(5).mean():
                    score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            range_high = df['High'].tail(40).max()
            range_low = df['Low'].tail(40).min()
            
            return {
                'detected': True,
                'pattern': 'Wyckoff Distribution',
                'signal': 'BEARISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Smart Money distribution. Buying climax → Range → UTAD → Markdown.',
                'entry_point': f"₹{range_low * 0.99:.2f} (SHORT below range)",
                'stop_loss': f"₹{range_high * 1.02:.2f} (Above UTAD high)",
                'target_1': f"₹{current_price * 0.85:.2f} (15% decline)",
                'target_2': f"₹{range_low - (range_high - range_low):.2f} (Range projected down)",
                'action': '🔻 SHORT on Phase E markdown with volume',
                'rules': [
                    'Phase A: Buying Climax detected',
                    'Phase B: Range at top',
                    'Phase C: UTAD completed',
                    'Phase D: Sign of Weakness',
                    f'SHORT Entry: Below ₹{range_low:.2f}',
                    'Volume on breakdown confirms'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    # ============================================================================
    # CANSLIM SCORING SYSTEM
    # ============================================================================
    
    def detect_canslim_setup(self) -> Dict:
        """
        Detect CANSLIM Pattern - William O'Neil's methodology.
        
        CANSLIM Criteria:
        - C: Current Quarterly Earnings (25%+ growth)
        - A: Annual Earnings Growth (25%+ for 3 years)
        - N: New Product/Management/High (52-week high)
        - S: Supply & Demand (Volume surge on breakout)
        - L: Leader or Laggard (Top 20% in industry)
        - I: Institutional Sponsorship (Funds buying)
        - M: Market Direction (Uptrend confirmed)
        """
        df = self.data
        if len(df) < 100:
            return {'detected': False, 'score': 0}
        
        score = 0
        max_score = 7  # One point per CANSLIM letter
        
        current_price = df['Close'].iloc[-1]
        
        # N: New High (within 15% of 52-week high)
        high_52w = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
        if current_price > high_52w * 0.85:
            score += 1
            if current_price >= high_52w * 0.98:  # Very close to new high
                score += 0.5
        
        # S: Supply & Demand (Volume analysis)
        recent_volume = df['Volume'].tail(20).mean()
        baseline_volume = df['Volume'].tail(100).mean()
        
        if recent_volume > baseline_volume * 1.5:  # Volume surge
            score += 1
        
        # Check for volume on up days vs down days
        last_20 = df.tail(20)
        up_days = last_20[last_20['Close'] > last_20['Open']]
        down_days = last_20[last_20['Close'] < last_20['Open']]
        
        if len(up_days) > 0 and len(down_days) > 0:
            if up_days['Volume'].mean() > down_days['Volume'].mean() * 1.2:
                score += 0.5
        
        # M: Market Direction (Price trend)
        sma_50 = df['Close'].tail(50).mean()
        sma_200 = df['Close'].tail(200).mean() if len(df) >= 200 else sma_50
        
        # Golden Cross or above both MAs
        if current_price > sma_50 and current_price > sma_200:
            score += 1
            if sma_50 > sma_200:  # Golden Cross
                score += 0.5
        
        # Price pattern: Cup with Handle or other base
        price_range_100 = (df['High'].tail(100).max() - df['Low'].tail(100).min()) / df['Close'].tail(100).mean()
        price_range_20 = (df['High'].tail(20).max() - df['Low'].tail(20).min()) / df['Close'].tail(20).mean()
        
        if price_range_100 > 0.20 and price_range_20 < 0.10:  # Base forming
            score += 1
        
        # Relative Strength (vs recent average)
        price_change_4w = (current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20] if len(df) >= 20 else 0
        if price_change_4w > 0.10:  # Outperforming (10%+ in 4 weeks)
            score += 1
        
        # Institutional accumulation proxy (Rising volume + rising price)
        last_30 = df.tail(30)
        volume_trend = np.polyfit(range(len(last_30)), last_30['Volume'].values, 1)[0]
        price_trend = np.polyfit(range(len(last_30)), last_30['Close'].values, 1)[0]
        
        if volume_trend > 0 and price_trend > 0:  # Both rising
            score += 1
        
        # Normalize score
        final_score = score / max_score
        
        if final_score > 0.65:  # At least 5 out of 7 criteria met
            pivot_point = df['High'].tail(20).max()
            base_low = df['Low'].tail(50).min()
            
            return {
                'detected': True,
                'pattern': 'CANSLIM Setup',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if final_score > 0.80 else 'MEDIUM',
                'score': final_score,
                'description': "William O'Neil's growth stock pattern. Multiple criteria aligned.",
                'entry_point': f"₹{pivot_point * 1.02:.2f} (Breakout above pivot)",
                'stop_loss': f"₹{base_low * 0.97:.2f} (Below base)",
                'target_1': f"₹{current_price * 1.20:.2f} (20% gain - typical first move)",
                'target_2': f"₹{current_price * 1.50:.2f} (50% gain - O'Neil target)",
                'action': 'BUY on breakout with 40-50% above average volume',
                'rules': [
                    f'CANSLIM Score: {final_score*100:.0f}%',
                    '✓ N: Near 52-week high' if current_price > high_52w * 0.85 else '✗ N: Not at new high',
                    '✓ S: Volume surge present' if recent_volume > baseline_volume * 1.5 else '✗ S: Low volume',
                    '✓ M: Above 50/200 SMA' if current_price > sma_50 and current_price > sma_200 else '✗ M: Below MAs',
                    f'Entry: Above ₹{pivot_point:.2f} with volume',
                    'Buy strongest stocks in strongest groups',
                    'Cut losses at 7-8% below entry'
                ]
            }
        
        return {'detected': False, 'score': final_score}
    
    # ============================================================================
    # ADDITIONAL ADVANCED PATTERNS
    # ============================================================================
    
    def detect_inverse_head_and_shoulders(self, lookback: int = 60) -> Dict:
        """
        Detect Inverse Head and Shoulders - Bullish reversal pattern.
        
        Pattern Characteristics:
        - Left shoulder (low), deeper head (lower low), right shoulder (low)
        - Neckline connects the two peaks between shoulders and head
        - Volume increasing through pattern (especially on right shoulder)
        - Breakout above neckline = BUY signal
        - Opposite of regular Head and Shoulders
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find local minima (troughs) and maxima (peaks)
        troughs = []
        peaks = []
        
        for i in range(5, len(prices) - 5):
            if prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6]):
                troughs.append((i, prices[i]))
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                peaks.append((i, prices[i]))
        
        # Need at least 3 troughs (shoulders + head) and 2 peaks (neckline points)
        if len(troughs) >= 3 and len(peaks) >= 2:
            # Check last 3 troughs for Inverse H&S pattern
            left_shoulder = troughs[-3]
            head = troughs[-2]
            right_shoulder = troughs[-1]
            
            # Head should be LOWER than both shoulders (deeper)
            if head[1] < left_shoulder[1] and head[1] < right_shoulder[1]:
                score += 0.3
                
                # Shoulders should be roughly equal (within 5%)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                if shoulder_diff < 0.05:
                    score += 0.2
                
                # Neckline (resistance level connecting peaks)
                if len(peaks) >= 2:
                    peak1 = peaks[-2]
                    peak2 = peaks[-1]
                    neckline = (peak1[1] + peak2[1]) / 2
                    
                    # Current price near or above neckline
                    current_price = prices[-1]
                    if current_price > neckline * 0.98:
                        score += 0.3
                    
                    # Volume increasing (bullish confirmation)
                    if df['Volume'].iloc[-10:].mean() > df['Volume'].iloc[-30:-10].mean():
                        score += 0.2
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            neckline = df['High'].tail(30).max()
            head_price = df['Low'].tail(40).min()
            pattern_height = neckline - head_price
            
            return {
                'detected': True,
                'pattern': 'Inverse Head and Shoulders',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Classic bullish reversal. Downtrend exhaustion turning bullish.',
                'entry_point': f"₹{neckline * 1.02:.2f} (BUY above neckline)",
                'stop_loss': f"₹{head_price * 0.98:.2f} (Below head)",
                'target_1': f"₹{neckline + pattern_height:.2f} (Height projected up)",
                'target_2': f"₹{neckline + (pattern_height * 1.5):.2f} (1.5x height up)",
                'action': 'BUY on neckline breakout with volume',
                'rules': [
                    'Wait for neckline break',
                    f'BUY Entry: Above ₹{neckline:.2f}',
                    f'Stop: Below ₹{head_price:.2f}',
                    'Volume surge on breakout',
                    'Pattern confirms trend reversal'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_triple_top(self, lookback: int = 60) -> Dict:
        """
        Detect Triple Top - Strong bearish reversal pattern.
        
        Pattern Characteristics:
        - Three peaks at similar price level (within 3%)
        - More reliable than double top
        - Troughs between peaks form support (neckline)
        - Breakdown below neckline = SHORT signal
        - High probability reversal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find local maxima (peaks)
        maxima_indices = []
        for i in range(3, len(prices) - 3):
            if prices[i] > max(prices[i-3:i]) and prices[i] > max(prices[i+1:i+4]):
                maxima_indices.append(i)
        
        if len(maxima_indices) >= 3:
            # Check last 3 peaks
            peak1_idx = maxima_indices[-3]
            peak2_idx = maxima_indices[-2]
            peak3_idx = maxima_indices[-1]
            
            peak1_price = prices[peak1_idx]
            peak2_price = prices[peak2_idx]
            peak3_price = prices[peak3_idx]
            
            # All three peaks should be within 3% of each other
            avg_peak = (peak1_price + peak2_price + peak3_price) / 3
            
            peak1_diff = abs(peak1_price - avg_peak) / avg_peak
            peak2_diff = abs(peak2_price - avg_peak) / avg_peak
            peak3_diff = abs(peak3_price - avg_peak) / avg_peak
            
            if peak1_diff < 0.03 and peak2_diff < 0.03 and peak3_diff < 0.03:
                score += 0.4
                
                # Find troughs between peaks
                trough1_section = prices[peak1_idx:peak2_idx]
                trough2_section = prices[peak2_idx:peak3_idx]
                
                if len(trough1_section) > 0 and len(trough2_section) > 0:
                    trough1 = np.min(trough1_section)
                    trough2 = np.min(trough2_section)
                    neckline = (trough1 + trough2) / 2
                    
                    # Volume declining through pattern
                    vol1 = df['Volume'].iloc[peak1_idx]
                    vol2 = df['Volume'].iloc[peak2_idx]
                    vol3 = df['Volume'].iloc[peak3_idx]
                    
                    if vol3 < vol2 and vol2 < vol1:
                        score += 0.2
                    
                    # Current price near or below neckline
                    current_price = prices[-1]
                    if current_price < neckline * 1.02:
                        score += 0.3
                    
                    # Volume increase on breakdown
                    recent_volume = df['Volume'].tail(3).mean()
                    avg_volume = df['Volume'].mean()
                    if recent_volume > avg_volume * 1.3:
                        score += 0.1
        
        if score > 0.7:
            top = df['High'].max()
            neckline = df['Low'].tail(40).min()
            pattern_height = top - neckline
            
            return {
                'detected': True,
                'pattern': 'Triple Top',
                'signal': 'BEARISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Strong bearish reversal. Three failed attempts at resistance.',
                'entry_point': f"₹{neckline * 0.98:.2f} (SHORT below neckline)",
                'stop_loss': f"₹{top * 1.02:.2f} (Above triple top)",
                'target_1': f"₹{neckline - pattern_height:.2f} (Height down)",
                'target_2': f"₹{neckline - (pattern_height * 1.5):.2f} (1.5x height down)",
                'action': '🔻 SHORT on neckline breakdown with volume',
                'rules': [
                    'Three tops within 3% range',
                    f'SHORT Entry: Below ₹{neckline:.2f}',
                    f'Stop: Above ₹{top:.2f}',
                    'Higher reliability than double top',
                    'Volume confirms breakdown'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_triple_bottom(self, lookback: int = 60) -> Dict:
        """
        Detect Triple Bottom - Strong bullish reversal pattern.
        
        Pattern Characteristics:
        - Three bottoms at similar price level (within 3%)
        - More reliable than double bottom
        - Peaks between bottoms form resistance (neckline)
        - Breakout above neckline = BUY signal
        - High probability reversal
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find local minima (troughs)
        minima_indices = []
        for i in range(3, len(prices) - 3):
            if prices[i] < min(prices[i-3:i]) and prices[i] < min(prices[i+1:i+4]):
                minima_indices.append(i)
        
        if len(minima_indices) >= 3:
            # Check last 3 bottoms
            trough1_idx = minima_indices[-3]
            trough2_idx = minima_indices[-2]
            trough3_idx = minima_indices[-1]
            
            trough1_price = prices[trough1_idx]
            trough2_price = prices[trough2_idx]
            trough3_price = prices[trough3_idx]
            
            # All three bottoms should be within 3% of each other
            avg_trough = (trough1_price + trough2_price + trough3_price) / 3
            
            trough1_diff = abs(trough1_price - avg_trough) / avg_trough
            trough2_diff = abs(trough2_price - avg_trough) / avg_trough
            trough3_diff = abs(trough3_price - avg_trough) / avg_trough
            
            if trough1_diff < 0.03 and trough2_diff < 0.03 and trough3_diff < 0.03:
                score += 0.4
                
                # Find peaks between troughs
                peak1_section = prices[trough1_idx:trough2_idx]
                peak2_section = prices[trough2_idx:trough3_idx]
                
                if len(peak1_section) > 0 and len(peak2_section) > 0:
                    peak1 = np.max(peak1_section)
                    peak2 = np.max(peak2_section)
                    neckline = (peak1 + peak2) / 2
                    
                    # Volume increasing through pattern (bullish)
                    vol1 = df['Volume'].iloc[trough1_idx]
                    vol2 = df['Volume'].iloc[trough2_idx]
                    vol3 = df['Volume'].iloc[trough3_idx]
                    
                    if vol3 > vol2 or vol2 > vol1:
                        score += 0.2
                    
                    # Current price near or above neckline
                    current_price = prices[-1]
                    if current_price > neckline * 0.98:
                        score += 0.3
                    
                    # Volume increase on breakout
                    recent_volume = df['Volume'].tail(3).mean()
                    avg_volume = df['Volume'].mean()
                    if recent_volume > avg_volume * 1.3:
                        score += 0.1
        
        if score > 0.7:
            bottom = df['Low'].min()
            neckline = df['High'].tail(40).max()
            pattern_height = neckline - bottom
            
            return {
                'detected': True,
                'pattern': 'Triple Bottom',
                'signal': 'BULLISH',
                'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                'score': score,
                'description': 'Strong bullish reversal. Three successful tests of support.',
                'entry_point': f"₹{neckline * 1.02:.2f} (BUY above neckline)",
                'stop_loss': f"₹{bottom * 0.98:.2f} (Below triple bottom)",
                'target_1': f"₹{neckline + pattern_height:.2f} (Height up)",
                'target_2': f"₹{neckline + (pattern_height * 1.5):.2f} (1.5x height up)",
                'action': 'BUY on neckline breakout with volume',
                'rules': [
                    'Three bottoms within 3% range',
                    f'BUY Entry: Above ₹{neckline:.2f}',
                    f'Stop: Below ₹{bottom:.2f}',
                    'Higher reliability than double bottom',
                    'Volume confirms breakout'
                ]
            }
        
        return {'detected': False, 'score': score}
    
    def detect_order_blocks(self, lookback: int = 50) -> Dict:
        """
        Detect Order Blocks - Smart Money / Institutional zones.
        
        Pattern Characteristics:
        - Last bullish/bearish candle before strong move
        - Represents institutional order accumulation
        - Price often returns to test these zones
        - High/Low of the block = support/resistance
        - Used by ICT (Inner Circle Trader) methodology
        
        Types:
        - Bullish Order Block: Last red candle before strong up move
        - Bearish Order Block: Last green candle before strong down move
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {'detected': False, 'score': 0}
        
        score = 0
        bullish_blocks = []
        bearish_blocks = []
        
        # Scan for order blocks
        for i in range(10, len(df) - 5):
            current_candle = df.iloc[i]
            next_5_candles = df.iloc[i+1:i+6]
            
            # Bullish Order Block: Last red candle before strong up move
            if current_candle['Close'] < current_candle['Open']:  # Red candle
                # Check if next 5 candles show strong upward movement
                future_gain = (next_5_candles['Close'].max() - current_candle['Close']) / current_candle['Close']
                
                if future_gain > 0.05:  # 5% up move after red candle
                    bullish_blocks.append({
                        'index': i,
                        'high': current_candle['High'],
                        'low': current_candle['Low'],
                        'close': current_candle['Close'],
                        'type': 'BULLISH',
                        'strength': future_gain
                    })
            
            # Bearish Order Block: Last green candle before strong down move
            elif current_candle['Close'] > current_candle['Open']:  # Green candle
                # Check if next 5 candles show strong downward movement
                future_loss = (current_candle['Close'] - next_5_candles['Close'].min()) / current_candle['Close']
                
                if future_loss > 0.05:  # 5% down move after green candle
                    bearish_blocks.append({
                        'index': i,
                        'high': current_candle['High'],
                        'low': current_candle['Low'],
                        'close': current_candle['Close'],
                        'type': 'BEARISH',
                        'strength': future_loss
                    })
        
        # Analyze most recent and strongest order blocks
        current_price = df['Close'].iloc[-1]
        
        # Find relevant order blocks near current price
        relevant_bullish = [b for b in bullish_blocks if abs(current_price - b['low']) / current_price < 0.10]
        relevant_bearish = [b for b in bearish_blocks if abs(current_price - b['high']) / current_price < 0.10]
        
        if relevant_bullish:
            # Sort by strength and recency
            relevant_bullish.sort(key=lambda x: (x['strength'], x['index']), reverse=True)
            strongest_bullish = relevant_bullish[0]
            
            # Check if price is near the bullish order block
            distance_to_block = abs(current_price - strongest_bullish['low']) / current_price
            
            if distance_to_block < 0.03:  # Within 3%
                score += 0.5
            elif distance_to_block < 0.05:  # Within 5%
                score += 0.3
            
            # Volume confirmation
            block_index = strongest_bullish['index']
            if block_index < len(df) - 1:
                block_volume = df['Volume'].iloc[block_index]
                avg_volume = df['Volume'].mean()
                
                if block_volume > avg_volume * 1.5:
                    score += 0.3
        
        if relevant_bearish:
            # Sort by strength and recency
            relevant_bearish.sort(key=lambda x: (x['strength'], x['index']), reverse=True)
            strongest_bearish = relevant_bearish[0]
            
            # Check if price is near the bearish order block
            distance_to_block = abs(current_price - strongest_bearish['high']) / current_price
            
            if distance_to_block < 0.03:  # Within 3%
                score += 0.5
            elif distance_to_block < 0.05:  # Within 5%
                score += 0.3
            
            # Volume confirmation
            block_index = strongest_bearish['index']
            if block_index < len(df) - 1:
                block_volume = df['Volume'].iloc[block_index]
                avg_volume = df['Volume'].mean()
                
                if block_volume > avg_volume * 1.5:
                    score += 0.3
        
        # Determine which signal to return
        if score > 0.6:
            if relevant_bullish and (not relevant_bearish or relevant_bullish[0]['index'] > relevant_bearish[0]['index']):
                # Bullish order block is more recent/relevant
                block = relevant_bullish[0]
                
                return {
                    'detected': True,
                    'pattern': 'Bullish Order Block',
                    'signal': 'BULLISH',
                    'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                    'score': score,
                    'description': 'Smart Money accumulation zone. Institutional buying detected.',
                    'entry_point': f"₹{block['low'] * 1.005:.2f} (Buy at order block low)",
                    'stop_loss': f"₹{block['low'] * 0.98:.2f} (Below order block)",
                    'target_1': f"₹{current_price * 1.08:.2f} (8% gain)",
                    'target_2': f"₹{current_price * 1.15:.2f} (15% gain)",
                    'action': 'BUY when price returns to order block zone',
                    'order_block_data': {
                        'high': block['high'],
                        'low': block['low'],
                        'type': 'BULLISH'
                    },
                    'rules': [
                        'Order block = institutional accumulation',
                        f'Support zone: ₹{block["low"]:.2f} - ₹{block["high"]:.2f}',
                        'Enter on retest of block',
                        'Stop below order block',
                        'ICT methodology: Smart Money Concepts'
                    ]
                }
            
            elif relevant_bearish:
                # Bearish order block
                block = relevant_bearish[0]
                
                return {
                    'detected': True,
                    'pattern': 'Bearish Order Block',
                    'signal': 'BEARISH',
                    'confidence': 'HIGH' if score > 0.8 else 'MEDIUM',
                    'score': score,
                    'description': 'Smart Money distribution zone. Institutional selling detected.',
                    'entry_point': f"₹{block['high'] * 0.995:.2f} (SHORT at order block high)",
                    'stop_loss': f"₹{block['high'] * 1.02:.2f} (Above order block)",
                    'target_1': f"₹{current_price * 0.92:.2f} (8% decline)",
                    'target_2': f"₹{current_price * 0.85:.2f} (15% decline)",
                    'action': '🔻 SHORT when price returns to order block zone',
                    'order_block_data': {
                        'high': block['high'],
                        'low': block['low'],
                        'type': 'BEARISH'
                    },
                    'rules': [
                        'Order block = institutional distribution',
                        f'Resistance zone: ₹{block["low"]:.2f} - ₹{block["high"]:.2f}',
                        'Enter on retest of block',
                        'Stop above order block',
                        'ICT methodology: Smart Money Concepts'
                    ]
                }
        
        return {'detected': False, 'score': score}
    
    def detect_elliott_wave(self, lookback: int = 100) -> Dict:
        """
        Detect Elliott Wave Pattern - Ralph Nelson Elliott's wave theory.
        
        Pattern Characteristics:
        - 5-wave impulse sequence (1-2-3-4-5) followed by 3-wave correction (A-B-C)
        - Wave 3 is never the shortest
        - Wave 2 never retraces more than 100% of Wave 1
        - Wave 4 does not overlap Wave 1 price territory
        - Wave 5 often shows divergence
        
        Simplified Detection:
        - Identify 5 distinct price swings in trend direction
        - Validate wave relationships
        - Detect completion of Wave 5 or correction
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 60:
            return {'detected': False, 'score': 0}
        
        score = 0
        prices = df['Close'].values
        
        # Find significant peaks and troughs
        swings = []
        
        for i in range(5, len(prices) - 5):
            # Peak detection
            if prices[i] == max(prices[i-5:i+6]):
                swings.append(('peak', i, prices[i]))
            # Trough detection
            elif prices[i] == min(prices[i-5:i+6]):
                swings.append(('trough', i, prices[i]))
        
        # Need at least 9 swings for 5-wave impulse + ABC correction
        # Or at least 5 swings for just impulse wave
        if len(swings) >= 9:
            # Analyze for complete pattern (5 waves + ABC)
            # Check last 9 swings for Elliott pattern
            
            # Simplified: Look for alternating pattern
            pattern_valid = True
            for i in range(len(swings) - 1):
                if swings[i][0] == swings[i+1][0]:  # Same type in sequence
                    pattern_valid = False
                    break
            
            if pattern_valid and len(swings) >= 9:
                # Extract potential waves (simplified)
                # In a bullish pattern: trough-peak-trough-peak-trough-peak-trough-peak-trough
                
                # Check for bullish impulse (starting with trough)
                if swings[-9][0] == 'trough':
                    wave1_start = swings[-9][2]
                    wave1_end = swings[-8][2]
                    wave2_end = swings[-7][2]
                    wave3_end = swings[-6][2]
                    wave4_end = swings[-5][2]
                    wave5_end = swings[-4][2]
                    
                    wave1_height = wave1_end - wave1_start
                    wave2_retrace = wave1_end - wave2_end
                    wave3_height = wave3_end - wave2_end
                    wave4_retrace = wave3_end - wave4_end
                    wave5_height = wave5_end - wave4_end
                    
                    # Elliott Wave Rules:
                    # 1. Wave 2 never retraces more than 100% of Wave 1
                    if wave2_retrace < wave1_height:
                        score += 0.2
                    
                    # 2. Wave 3 is never the shortest
                    if wave3_height >= wave1_height and wave3_height >= wave5_height:
                        score += 0.2
                    
                    # 3. Wave 4 does not overlap Wave 1
                    if wave4_end > wave1_end:
                        score += 0.2
                    
                    # 4. Wave 3 usually extends (is longest)
                    if wave3_height > wave1_height and wave3_height > wave5_height:
                        score += 0.1
                    
                    # 5. Check if we're at Wave 5 completion (potential reversal)
                    current_price = prices[-1]
                    if abs(current_price - wave5_end) / wave5_end < 0.05:
                        score += 0.2
                        
                        # Volume divergence at Wave 5
                        wave5_idx = swings[-4][1]
                        wave3_idx = swings[-6][1]
                        
                        if wave5_idx < len(df) and wave3_idx < len(df):
                            vol_wave5 = df['Volume'].iloc[wave5_idx]
                            vol_wave3 = df['Volume'].iloc[wave3_idx]
                            
                            if vol_wave5 < vol_wave3:  # Divergence
                                score += 0.1
        
        elif len(swings) >= 5:
            # Check for impulse wave only (5 waves)
            if swings[-5][0] == 'trough':  # Bullish impulse
                score += 0.3
        
        if score > 0.7:
            current_price = df['Close'].iloc[-1]
            
            # Determine wave position
            wave_position = "Wave 5 completion" if len(swings) >= 9 else "Impulse wave detected"
            
            # For completed Wave 5, expect ABC correction
            if len(swings) >= 9:
                wave5_price = swings[-4][2]
                wave4_price = swings[-5][2]
                
                # ABC correction targets
                correction_38 = wave5_price - (wave5_price - wave4_price) * 0.382
                correction_50 = wave5_price - (wave5_price - wave4_price) * 0.50
                correction_62 = wave5_price - (wave5_price - wave4_price) * 0.618
                
                return {
                    'detected': True,
                    'pattern': 'Elliott Wave (5-3 Pattern)',
                    'signal': 'NEUTRAL',  # Can be bullish or bearish depending on position
                    'confidence': 'MEDIUM',
                    'score': score,
                    'description': f'{wave_position}. 5-wave impulse complete, expect 3-wave correction.',
                    'entry_point': f"₹{correction_50:.2f} (Enter at 50% correction - Wave C target)",
                    'stop_loss': f"₹{correction_62 * 0.98:.2f} (Below 61.8% correction)",
                    'target_1': f"₹{wave5_price * 1.10:.2f} (Next impulse wave)",
                    'target_2': f"₹{wave5_price * 1.20:.2f} (Extended target)",
                    'action': 'WAIT for ABC correction, then BUY at Wave C completion',
                    'elliott_data': {
                        'wave_position': wave_position,
                        'correction_levels': {
                            '38.2%': correction_38,
                            '50%': correction_50,
                            '61.8%': correction_62
                        }
                    },
                    'rules': [
                        'Elliott Wave: 5-3 pattern (5 impulse + 3 correction)',
                        f'Wave 5 complete at ₹{wave5_price:.2f}',
                        'Expect ABC correction now',
                        f'Buy at C wave: ₹{correction_50:.2f} (50% retracement)',
                        'Wave 3 never shortest',
                        'Wave 4 never overlaps Wave 1',
                        'Professional institutional pattern'
                    ]
                }
            else:
                # Mid-impulse wave
                return {
                    'detected': True,
                    'pattern': 'Elliott Wave (Impulse)',
                    'signal': 'BULLISH',
                    'confidence': 'MEDIUM',
                    'score': score,
                    'description': 'Elliott impulse wave in progress. Continuation expected.',
                    'entry_point': f"₹{current_price * 1.02:.2f} (Current wave continuation)",
                    'stop_loss': f"₹{current_price * 0.95:.2f} (Below recent low)",
                    'target_1': f"₹{current_price * 1.15:.2f} (Wave extension)",
                    'target_2': f"₹{current_price * 1.30:.2f} (Full impulse target)",
                    'action': 'BUY on pullbacks within impulse wave',
                    'rules': [
                        'Elliott impulse wave detected',
                        'Trend continuation expected',
                        'Wait for Wave 2/4 pullbacks to enter',
                        'Target Wave 5 completion'
                    ]
                }
        
        return {'detected': False, 'score': score}
    
    def detect_mean_reversion(self, lookback: int = 50) -> Dict:
        """
        Detect Mean Reversion Setup - Statistical trading strategy.
        
        Pattern Characteristics:
        - Price extended from moving average (2+ standard deviations)
        - Bollinger Bands squeeze or extreme stretch
        - RSI oversold (<30) or overbought (>70)
        - Volume spike on exhaustion
        - Reversion to mean expected
        
        Types:
        - Bullish Reversion: Oversold, far below mean
        - Bearish Reversion: Overbought, far above mean
        
        Strategy:
        - Enter when price is 2+ standard deviations from mean
        - Exit at mean (moving average)
        - Works best in ranging/sideways markets
        """
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {'detected': False, 'score': 0}
        
        score = 0
        current_price = df['Close'].iloc[-1]
        
        # Calculate statistical measures
        sma_20 = df['Close'].tail(20).mean()
        std_20 = df['Close'].tail(20).std()
        
        # Bollinger Bands (if available, otherwise calculate)
        if 'BB_High' in df.columns and 'BB_Low' in df.columns and 'BB_Mid' in df.columns:
            bb_upper = df['BB_High'].iloc[-1]
            bb_lower = df['BB_Low'].iloc[-1]
            bb_mid = df['BB_Mid'].iloc[-1]
        else:
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            bb_mid = sma_20
        
        # Calculate distance from mean in standard deviations
        distance_from_mean = (current_price - sma_20) / std_20 if std_20 > 0 else 0
        
        # RSI (if available)
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        
        # Determine if bullish or bearish reversion setup
        is_bullish_reversion = False
        is_bearish_reversion = False
        
        # Bullish Mean Reversion (Oversold)
        if distance_from_mean < -1.5:  # 1.5+ std below mean
            is_bullish_reversion = True
            score += 0.3
            
            # Beyond 2 std is stronger
            if distance_from_mean < -2.0:
                score += 0.2
            
            # Price below lower Bollinger Band
            if current_price < bb_lower:
                score += 0.2
            
            # RSI oversold
            if rsi < 30:
                score += 0.2
            elif rsi < 40:
                score += 0.1
            
            # Volume spike (panic selling)
            recent_volume = df['Volume'].tail(3).mean()
            avg_volume = df['Volume'].mean()
            
            if recent_volume > avg_volume * 1.5:
                score += 0.1
        
        # Bearish Mean Reversion (Overbought)
        elif distance_from_mean > 1.5:  # 1.5+ std above mean
            is_bearish_reversion = True
            score += 0.3
            
            # Beyond 2 std is stronger
            if distance_from_mean > 2.0:
                score += 0.2
            
            # Price above upper Bollinger Band
            if current_price > bb_upper:
                score += 0.2
            
            # RSI overbought
            if rsi > 70:
                score += 0.2
            elif rsi > 60:
                score += 0.1
            
            # Volume spike (buying exhaustion)
            recent_volume = df['Volume'].tail(3).mean()
            avg_volume = df['Volume'].mean()
            
            if recent_volume > avg_volume * 1.5:
                score += 0.1
        
        # Additional confirmation: Market in range (not trending)
        # Check if price has been oscillating
        highs_20 = df['High'].tail(20)
        lows_20 = df['Low'].tail(20)
        range_pct = (highs_20.max() - lows_20.min()) / sma_20
        
        # Prefer mean reversion in ranging markets (not strong trends)
        if 0.10 < range_pct < 0.30:  # 10-30% range
            score += 0.1
        
        if score > 0.7:
            # Bullish Reversion Setup
            if is_bullish_reversion:
                target_mean = sma_20
                stop_loss = current_price * 0.95  # 5% hard stop
                
                # Calculate expected move back to mean
                reversion_gain = (target_mean - current_price) / current_price
                
                return {
                    'detected': True,
                    'pattern': 'Mean Reversion (Bullish)',
                    'signal': 'BULLISH',
                    'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                    'score': score,
                    'description': f'Oversold reversion setup. Price {abs(distance_from_mean):.1f} std below mean.',
                    'entry_point': f"₹{current_price:.2f} (Current - enter on extreme)",
                    'stop_loss': f"₹{stop_loss:.2f} (5% below entry)",
                    'target_1': f"₹{bb_mid:.2f} (Mean / Middle BB)",
                    'target_2': f"₹{bb_upper:.2f} (Upper BB)",
                    'action': 'BUY oversold extreme, exit at mean',
                    'mean_reversion_data': {
                        'sma_20': sma_20,
                        'std_deviation': distance_from_mean,
                        'bb_lower': bb_lower,
                        'bb_mid': bb_mid,
                        'bb_upper': bb_upper,
                        'rsi': rsi,
                        'expected_gain': reversion_gain * 100
                    },
                    'rules': [
                        f'Price: {abs(distance_from_mean):.1f} std below mean',
                        f'Current: ₹{current_price:.2f}',
                        f'Mean (SMA-20): ₹{sma_20:.2f}',
                        f'RSI: {rsi:.1f} (Oversold)' if rsi < 40 else f'RSI: {rsi:.1f}',
                        f'Expected reversion: {reversion_gain*100:.1f}%',
                        'Exit at mean or resistance',
                        'Works best in ranging markets',
                        'Statistical edge: 2+ std deviations'
                    ]
                }
            
            # Bearish Reversion Setup
            else:  # is_bearish_reversion
                target_mean = sma_20
                stop_loss = current_price * 1.05  # 5% hard stop
                
                # Calculate expected move back to mean
                reversion_decline = (current_price - target_mean) / current_price
                
                return {
                    'detected': True,
                    'pattern': 'Mean Reversion (Bearish)',
                    'signal': 'BEARISH',
                    'confidence': 'HIGH' if score > 0.85 else 'MEDIUM',
                    'score': score,
                    'description': f'Overbought reversion setup. Price {abs(distance_from_mean):.1f} std above mean.',
                    'entry_point': f"₹{current_price:.2f} (SHORT current - enter on extreme)",
                    'stop_loss': f"₹{stop_loss:.2f} (5% above entry)",
                    'target_1': f"₹{bb_mid:.2f} (Mean / Middle BB)",
                    'target_2': f"₹{bb_lower:.2f} (Lower BB)",
                    'action': '🔻 SHORT overbought extreme, cover at mean',
                    'mean_reversion_data': {
                        'sma_20': sma_20,
                        'std_deviation': distance_from_mean,
                        'bb_lower': bb_lower,
                        'bb_mid': bb_mid,
                        'bb_upper': bb_upper,
                        'rsi': rsi,
                        'expected_decline': reversion_decline * 100
                    },
                    'rules': [
                        f'Price: {abs(distance_from_mean):.1f} std above mean',
                        f'Current: ₹{current_price:.2f}',
                        f'Mean (SMA-20): ₹{sma_20:.2f}',
                        f'RSI: {rsi:.1f} (Overbought)' if rsi > 60 else f'RSI: {rsi:.1f}',
                        f'Expected reversion: {reversion_decline*100:.1f}%',
                        'Cover at mean or support',
                        'Works best in ranging markets',
                        'Statistical edge: 2+ std deviations'
                    ]
                }
        
        return {'detected': False, 'score': score}
    
    # ============================================================================
    # MAIN DETECTION METHODS
    # ============================================================================
    
    def detect_all_zanger_patterns(self) -> List[Dict]:
        """
        Detect all Dan Zanger patterns.
        
        Returns:
            List of detected patterns with full details
        """
        patterns = []
        
        # Run all Dan Zanger pattern detections
        detectors = [
            self.detect_cup_and_handle,
            self.detect_high_tight_flag,
            self.detect_ascending_triangle,
            self.detect_flat_base,
            self.detect_falling_wedge,
            self.detect_double_bottom
        ]
        
        for detector in detectors:
            result = detector()
            if result.get('detected', False):
                patterns.append(result)
        
        return patterns
    
    def detect_all_classic_patterns(self) -> List[Dict]:
        """
        Detect all classic chart patterns (including bearish patterns).
        
        Returns:
            List of detected patterns with full details
        """
        patterns = []
        
        # Run all classic pattern detections
        detectors = [
            self.detect_head_and_shoulders,
            self.detect_double_top,
            self.detect_descending_triangle,
            self.detect_symmetrical_triangle,
            self.detect_bull_flag,
            self.detect_bear_flag,
            self.detect_rising_wedge,
            self.detect_pennant
        ]
        
        for detector in detectors:
            result = detector()
            if result.get('detected', False):
                patterns.append(result)
        
        return patterns
    
    def detect_all_swing_patterns(self) -> List[Dict]:
        """
        Detect all Qullamaggie swing patterns.
        
        Returns:
            List of detected patterns with full details
        """
        patterns = []
        
        # Run all Qullamaggie pattern detections
        detectors = [
            self.detect_qullamaggie_breakout,
            self.detect_episodic_pivot,
            self.detect_parabolic_short,
            self.detect_gap_and_go,
            self.detect_abcd_pattern
        ]
        
        for detector in detectors:
            result = detector()
            if result.get('detected', False):
                patterns.append(result)
        
        return patterns
    
    def detect_all_wyckoff_canslim_patterns(self) -> List[Dict]:
        """
        Detect Wyckoff, CANSLIM, VCP, Darvas Box, and advanced patterns.
        
        Returns:
            List of detected patterns with full details
        """
        patterns = []
        
        # Run all advanced pattern detections
        detectors = [
            self.detect_vcp,
            self.detect_darvas_box,
            self.detect_wyckoff_accumulation,
            self.detect_wyckoff_distribution,
            self.detect_canslim_setup,
            self.detect_inverse_head_and_shoulders,
            self.detect_triple_top,
            self.detect_triple_bottom,
            self.detect_order_blocks,
            self.detect_elliott_wave,
            self.detect_mean_reversion
        ]
        
        for detector in detectors:
            result = detector()
            if result.get('detected', False):
                patterns.append(result)
        
        return patterns
    
    def detect_all_patterns(self) -> Dict[str, List[Dict]]:
        """
        Detect all patterns across all categories.
        
        Returns:
            Dictionary with pattern categories and detected patterns
        """
        zanger = self.detect_all_zanger_patterns()
        classic = self.detect_all_classic_patterns()
        swing = self.detect_all_swing_patterns()
        wyckoff_canslim = self.detect_all_wyckoff_canslim_patterns()
        
        return {
            'zanger_patterns': zanger,
            'classic_patterns': classic,
            'swing_patterns': swing,
            'wyckoff_canslim_patterns': wyckoff_canslim,
            'all_patterns': zanger + classic + swing + wyckoff_canslim
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_pattern_summary(pattern: Dict) -> str:
    """
    Format pattern details into readable summary.
    
    Args:
        pattern: Pattern dictionary
        
    Returns:
        Formatted string summary
    """
    if not pattern.get('detected', False):
        return "No pattern detected"
    
    summary = f"""
    Pattern: {pattern['pattern']}
    Signal: {pattern['signal']}
    Confidence: {pattern['confidence']}
    Score: {pattern['score']:.2f}
    
    Description: {pattern['description']}
    Action: {pattern['action']}
    
    Entry: {pattern.get('entry_point', 'N/A')}
    Stop Loss: {pattern.get('stop_loss', 'N/A')}
    Target 1: {pattern.get('target_1', 'N/A')}
    Target 2: {pattern.get('target_2', 'N/A')}
    
    Rules:
    """
    
    for rule in pattern.get('rules', []):
        summary += f"\n    - {rule}"
    
    return summary


def get_pattern_statistics(patterns: List[Dict]) -> Dict:
    """
    Calculate statistics across detected patterns.
    
    Args:
        patterns: List of detected patterns
        
    Returns:
        Dictionary with pattern statistics
    """
    if not patterns:
        return {
            'total_patterns': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_score': 0,
            'high_confidence_count': 0
        }
    
    bullish = sum(1 for p in patterns if p.get('signal') == 'BULLISH')
    bearish = sum(1 for p in patterns if p.get('signal') == 'BEARISH')
    neutral = sum(1 for p in patterns if p.get('signal') in ['NEUTRAL', 'REVERSAL'])
    
    scores = [p.get('score', 0) for p in patterns]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    high_conf = sum(1 for p in patterns if p.get('confidence') == 'HIGH')
    
    return {
        'total_patterns': len(patterns),
        'bullish_count': bullish,
        'bearish_count': bearish,
        'neutral_count': neutral,
        'avg_score': avg_score,
        'high_confidence_count': high_conf
    }


# Example usage
if __name__ == "__main__":
    print("Pattern Detector Module - Ready for import")
    print("Available classes: PatternDetector")
    print("Available functions: format_pattern_summary, get_pattern_statistics")
    print("\nTotal Patterns: 30")
    print("- Zanger: 6")
    print("- Classic: 8 (including bearish with SHORT signals)")
    print("- Swing: 5")
    print("- Advanced: 11 (VCP, Darvas, Wyckoff, CANSLIM, Inv H&S, Triple Top/Bottom, Order Blocks, Elliott Wave, Mean Reversion)")
