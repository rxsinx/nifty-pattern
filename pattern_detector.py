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
    
    def detect_all_patterns(self) -> Dict[str, List[Dict]]:
        """
        Detect all patterns across all categories.
        
        Returns:
            Dictionary with pattern categories and detected patterns
        """
        zanger = self.detect_all_zanger_patterns()
        classic = self.detect_all_classic_patterns()
        swing = self.detect_all_swing_patterns()
        
        return {
            'zanger_patterns': zanger,
            'classic_patterns': classic,
            'swing_patterns': swing,
            'all_patterns': zanger + classic + swing
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
    print("\nTotal Patterns: 19")
    print("- Zanger: 6")
    print("- Classic: 8 (including bearish with SHORT signals)")
    print("- Swing: 5")
