"""
Pattern Detection Module for Technical Analysis
================================================

This module contains all chart pattern detection algorithms including:
- Dan Zanger's patterns (Cup and Handle, High Tight Flag, etc.)
- Qullamaggie's swing patterns (Breakout, Episodic Pivot, etc.)
- Classic chart patterns (Double Bottom, Falling Wedge, etc.)

Author: Indian Equity Analyzer Pro
Version: 3.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

class PatternConfidence(Enum):
    """Pattern confidence levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class PatternSignal(Enum):
    """Pattern trading signals"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    REVERSAL = "REVERSAL"

@dataclass
class PatternResult:
    """Standardized pattern result container"""
    detected: bool
    pattern_name: str
    signal: PatternSignal
    confidence: PatternConfidence
    score: float
    description: str
    action: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    risk_reward: Optional[float] = None
    rules: List[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility"""
        return {
            'detected': self.detected,
            'pattern': self.pattern_name,
            'signal': self.signal.value,
            'confidence': self.confidence.value,
            'score': self.score,
            'description': self.description,
            'action': self.action,
            'entry_point': f"₹{self.entry_price:.2f}" if self.entry_price else "N/A",
            'stop_loss': f"₹{self.stop_loss:.2f}" if self.stop_loss else "N/A",
            'target_1': f"₹{self.target_1:.2f}" if self.target_1 else "N/A",
            'target_2': f"₹{self.target_2:.2f}" if self.target_2 else "N/A",
            'target_3': f"₹{self.target_3:.2f}" if self.target_3 else "N/A",
            'risk_reward': f"1:{self.risk_reward:.2f}" if self.risk_reward else "N/A",
            'rules': self.rules or [],
            'metadata': self.metadata or {}
        }


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
        self.data = data.copy()
        self._validate_data()
        self._enhance_data()
        self.pattern_history: List[PatternResult] = []
        
        # Configuration
        self.min_data_points = 20
        self.volume_spike_threshold = 2.5  # 2.5x average volume
        self.volume_dry_up_threshold = 0.6  # 40% reduction
        
    def _validate_data(self) -> None:
        """Validate input data"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(self.data) < self.min_data_points:
            warnings.warn(f"Limited data: only {len(self.data)} periods available")
    
    def _enhance_data(self) -> None:
        """Enhance data with additional calculated fields"""
        # Ensure EMAs are present for Qullamaggie patterns
        if 'EMA_8' not in self.data.columns:
            self.data['EMA_8'] = self.data['Close'].ewm(span=8, adjust=False).mean()
        if 'EMA_21' not in self.data.columns:
            self.data['EMA_21'] = self.data['Close'].ewm(span=21, adjust=False).mean()
        
        # Calculate additional useful indicators
        if 'ATR' not in self.data.columns:
            # Simple ATR approximation
            high_low = self.data['High'] - self.data['Low']
            high_close = np.abs(self.data['High'] - self.data['Close'].shift())
            low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.data['ATR'] = tr.rolling(window=14).mean()
        
        # Volume moving average
        if 'Volume_SMA' not in self.data.columns:
            self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # Price returns and volatility
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume distribution for pattern confirmation"""
        if len(df) < 20:
            return {}
        
        # Calculate volume at price levels
        price_min = df['Low'].min()
        price_max = df['High'].max()
        
        if price_max <= price_min:
            price_max = price_min * 1.01
        
        num_bins = min(30, max(10, len(df) // 3))
        bins = np.linspace(price_min, price_max, num_bins)
        
        volume_by_price = []
        for i in range(len(bins) - 1):
            mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
            volume_by_price.append(df.loc[mask, 'Volume'].sum() if mask.any() else 0)
        
        volume_by_price = np.array(volume_by_price)
        
        if len(volume_by_price) == 0 or np.sum(volume_by_price) == 0:
            return {}
        
        # Find Point of Control (POC)
        poc_idx = np.argmax(volume_by_price)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Find high volume nodes (top 20%)
        if len(volume_by_price) > 0:
            threshold = np.percentile(volume_by_price[volume_by_price > 0], 80)
            high_volume_nodes = bins[:-1][volume_by_price > threshold]
        else:
            high_volume_nodes = []
        
        return {
            'poc_price': poc_price,
            'high_volume_nodes': high_volume_nodes,
            'volume_distribution': volume_by_price
        }
    
    def _calculate_returns_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate returns distribution statistics"""
        returns = df['Returns'].dropna()
        
        if len(returns) < 10:
            return {}
        
        return {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'positive_ratio': (returns > 0).sum() / len(returns)
        }
    
    # ============================================================================
    # DAN ZANGER PATTERNS
    # ============================================================================
    
    def detect_cup_and_handle(self, lookback: int = 100) -> PatternResult:
        """
        Detect Cup and Handle pattern - Dan Zanger's signature pattern.
        
        Pattern Characteristics:
        - U-shaped cup formation (7-8 weeks minimum)
        - Handle in upper half of cup (1-4 weeks)
        - Volume dry-up in handle
        - Breakout with 3x+ volume
        """
        df = self.data.tail(lookback).copy()
        
        # Minimum requirements check
        if len(df) < 60:
            return PatternResult(
                detected=False,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data for pattern detection",
                action="Wait for more data",
                rules=["Minimum 60 periods required"]
            )
        
        score = 0.0
        prices = df['Close'].values
        
        # Find the cup (minimum price point)
        min_price_idx = np.argmin(prices)
        
        # Validation: cup should not be at extremes
        if min_price_idx < 15 or min_price_idx > len(prices) - 15:
            return PatternResult(
                detected=False,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Cup bottom at extreme position",
                action="Not a valid cup formation",
                rules=["Cup bottom should be in middle third of data"]
            )
        
        # Cup formation analysis
        cup_left = prices[:min_price_idx]
        cup_right = prices[min_price_idx:]
        
        if len(cup_left) < 20 or len(cup_right) < 20:
            return PatternResult(
                detected=False,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient cup formation",
                action="Wait for clearer pattern",
                rules=["Cup requires at least 20 periods on each side"]
            )
        
        # Cup depth analysis (ideal: 15-40%)
        left_peak = cup_left[0]
        right_peak = cup_right[-1] if len(cup_right) > 0 else cup_left[-1]
        cup_top = max(left_peak, right_peak)
        cup_bottom = prices[min_price_idx]
        
        cup_depth = (cup_top - cup_bottom) / cup_top
        
        if 0.12 <= cup_depth <= 0.45:
            if 0.15 <= cup_depth <= 0.35:
                score += 0.25  # Ideal range
            else:
                score += 0.15  # Acceptable but not ideal
        else:
            return PatternResult(
                detected=False,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description=f"Cup depth {cup_depth:.1%} outside ideal range (12-45%)",
                action="Not a valid cup formation",
                rules=[f"Cup depth: {cup_depth:.1%}"]
            )
        
        # Handle formation (last 15-25% of data)
        handle_start = int(len(df) * 0.75)
        handle_data = df.iloc[handle_start:]
        
        if len(handle_data) < 8:
            return PatternResult(
                detected=False,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Handle too short",
                action="Wait for handle formation",
                rules=["Handle requires at least 8 periods"]
            )
        
        # Handle should be in upper half of cup
        cup_mid = (cup_top + cup_bottom) / 2
        handle_avg = handle_data['Close'].mean()
        
        if handle_avg > cup_mid:
            score += 0.20
        elif handle_avg > cup_bottom * 1.1:
            score += 0.10
        else:
            return PatternResult(
                detected=False,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=score,
                description="Handle in lower half of cup",
                action="Not a valid handle formation",
                rules=["Handle must be in upper half of cup"]
            )
        
        # Volume analysis
        volume_analysis = self._analyze_volume_profile(df)
        
        # Volume dry-up in handle
        cup_volume = df['Volume'].iloc[:handle_start].mean()
        handle_volume = handle_data['Volume'].mean()
        
        if handle_volume < cup_volume * self.volume_dry_up_threshold:
            score += 0.20
        elif handle_volume < cup_volume * 0.8:
            score += 0.10
        
        # Handle tightness (should be consolidating)
        handle_range = (handle_data['High'].max() - handle_data['Low'].min()) / handle_avg
        if handle_range < 0.12:
            score += 0.20
        elif handle_range < 0.18:
            score += 0.10
        
        # U-shape validation
        left_trend = np.polyfit(range(len(cup_left)), cup_left, 1)[0]
        right_trend = np.polyfit(range(len(cup_right)), cup_right, 1)[0]
        
        if left_trend < 0 and right_trend > 0:
            score += 0.15
        elif left_trend < 0 or right_trend > 0:
            score += 0.05
        
        # Time analysis (assuming daily data)
        cup_duration = min_price_idx  # Periods in cup
        handle_duration = len(handle_data)
        
        # Cup: 35-65 days (7-13 weeks)
        if 35 <= cup_duration <= 65:
            score += 0.10
        elif 25 <= cup_duration <= 75:
            score += 0.05
        
        # Handle: 5-20 days (1-4 weeks)
        if 5 <= handle_duration <= 20:
            score += 0.10
        elif 3 <= handle_duration <= 25:
            score += 0.05
        
        # Pattern detection threshold
        if score >= 0.75:
            current_price = df['Close'].iloc[-1]
            handle_high = handle_data['High'].max()
            handle_low = handle_data['Low'].min()
            
            # Calculate entry, stop, targets
            entry_price = handle_high * 1.01
            stop_loss = handle_low * 0.98
            target_1 = current_price * 1.15
            target_2 = current_price + (cup_top - cup_bottom)  # Cup depth projected
            target_3 = target_2 * 1.15
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Cup and Handle",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Most powerful bull market pattern. U-shaped cup with handle in upper half showing accumulation.",
                action="BUY on breakout above handle high with >3x volume confirmation",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Cup depth: {cup_depth:.1%} (ideal: 15-35%)",
                    f"Handle position: {((handle_avg - cup_bottom)/(cup_top - cup_bottom)):.0%} of cup height",
                    f"Entry: Above ₹{handle_high:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f} ({(entry_price - stop_loss)/entry_price:.1%})",
                    f"Volume requirement: 3x average on breakout",
                    f"Timeframe: Cup {cup_duration} days, Handle {handle_duration} days"
                ],
                metadata={
                    'cup_depth': cup_depth,
                    'cup_duration': cup_duration,
                    'handle_duration': handle_duration,
                    'handle_position': ((handle_avg - cup_bottom)/(cup_top - cup_bottom)),
                    'volume_profile': volume_analysis
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Cup and Handle",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Pattern detected but below confidence threshold",
            action="Monitor for pattern development",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.75)"]
        )
    
    def detect_high_tight_flag(self, lookback: int = 40) -> PatternResult:
        """
        Detect High Tight Flag pattern - Explosive continuation pattern.
        
        Pattern Characteristics:
        - Strong pole (>20% gain in <4 weeks)
        - Tight flag (<15% of pole height)
        - Volume dry-up during flag
        - Flag above midpoint of pole
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 30:
            return PatternResult(
                detected=False,
                pattern_name="High Tight Flag",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 30 periods required"]
            )
        
        score = 0.0
        
        # Identify pole (strong uptrend) - first 40-60% of data
        pole_length = min(20, len(df) // 2)
        pole_data = df.head(pole_length)
        
        if len(pole_data) < 10:
            return PatternResult(
                detected=False,
                pattern_name="High Tight Flag",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Pole too short",
                action="Not a valid pole formation",
                rules=["Pole requires at least 10 periods"]
            )
        
        pole_gain = (pole_data['Close'].iloc[-1] - pole_data['Close'].iloc[0]) / pole_data['Close'].iloc[0]
        
        # Pole gain requirements
        if pole_gain > 0.25:
            score += 0.30  # Excellent pole
        elif pole_gain > 0.18:
            score += 0.20  # Good pole
        else:
            return PatternResult(
                detected=False,
                pattern_name="High Tight Flag",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description=f"Pole gain {pole_gain:.1%} insufficient (need >18%)",
                action="Not a valid pole",
                rules=[f"Pole gain: {pole_gain:.1%}"]
            )
        
        # Flag consolidation (remaining data)
        flag_data = df.tail(len(df) - pole_length)
        
        if len(flag_data) < 8:
            return PatternResult(
                detected=False,
                pattern_name="High Tight Flag",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Flag too short",
                action="Wait for flag formation",
                rules=["Flag requires at least 8 periods"]
            )
        
        flag_range = (flag_data['High'].max() - flag_data['Low'].min()) / flag_data['Close'].mean()
        
        # Flag tightness
        if flag_range < 0.12:
            score += 0.25  # Very tight flag
        elif flag_range < 0.18:
            score += 0.15  # Acceptable flag
        else:
            return PatternResult(
                detected=False,
                pattern_name="High Tight Flag",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=score,
                description=f"Flag too wide: {flag_range:.1%} (need <18%)",
                action="Not a tight flag formation",
                rules=[f"Flag range: {flag_range:.1%}"]
            )
        
        # Volume analysis
        pole_volume = pole_data['Volume'].mean()
        flag_volume = flag_data['Volume'].mean()
        
        if flag_volume < pole_volume * self.volume_dry_up_threshold:
            score += 0.20
        elif flag_volume < pole_volume * 0.8:
            score += 0.10
        
        # Flag position relative to pole
        pole_mid = (pole_data['Close'].iloc[0] + pole_data['Close'].iloc[-1]) / 2
        flag_avg = flag_data['Close'].mean()
        
        if flag_avg > pole_mid:
            score += 0.15  # Flag in upper half
        elif flag_avg > pole_data['Close'].iloc[0] * 1.1:
            score += 0.10  # Flag above pole start
        
        # Flag duration (should be shorter than pole)
        if len(flag_data) < len(pole_data) * 0.8:
            score += 0.10
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            pole_high = df['High'].head(pole_length).max()
            flag_low = flag_data['Low'].min()
            pole_height = pole_high - pole_data['Close'].iloc[0]
            
            # Calculate entry, stop, targets
            entry_price = pole_high * 1.02
            stop_loss = flag_low * 0.97
            target_1 = pole_high + pole_height  # Pole height projected
            target_2 = target_1 * 1.15
            target_3 = target_1 * 1.30
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="High Tight Flag",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Rare explosive continuation pattern. Strong pole followed by tight consolidation flag.",
                action="BUY on breakout with massive volume (>5x average)",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Pole gain: {pole_gain:.1%} in {pole_length} periods",
                    f"Flag range: {flag_range:.1%} of price",
                    f"Entry: Above ₹{pole_high:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f}",
                    f"Target: Pole height projection + extensions",
                    f"Volume: Must surge on breakout"
                ],
                metadata={
                    'pole_gain': pole_gain,
                    'pole_length': pole_length,
                    'flag_range': flag_range,
                    'flag_duration': len(flag_data),
                    'volume_ratio': flag_volume / pole_volume if pole_volume > 0 else 0
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="High Tight Flag",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Pattern elements present but below confidence threshold",
            action="Monitor for breakout confirmation",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_ascending_triangle(self, lookback: int = 40) -> PatternResult:
        """
        Detect Ascending Triangle - Bullish continuation pattern.
        
        Pattern Characteristics:
        - Flat resistance (horizontal top)
        - Rising support (higher lows)
        - Volume declining during formation
        - Breakout with volume surge
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 25:
            return PatternResult(
                detected=False,
                pattern_name="Ascending Triangle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 25 periods required"]
            )
        
        score = 0.0
        
        # Resistance line analysis (should be flat)
        highs = df['High'].values
        
        # Calculate resistance flatness
        resistance_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        resistance_variance = np.std(highs) / np.mean(highs)
        
        if resistance_variance < 0.025 and abs(resistance_slope) < 0.0005:
            score += 0.30  # Excellent flat resistance
        elif resistance_variance < 0.035 and abs(resistance_slope) < 0.001:
            score += 0.20  # Good resistance
        else:
            return PatternResult(
                detected=False,
                pattern_name="Ascending Triangle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description=f"Resistance not flat enough (variance: {resistance_variance:.1%})",
                action="Not a valid triangle",
                rules=[f"Resistance variance: {resistance_variance:.1%}"]
            )
        
        # Support line analysis (should be rising)
        lows = df['Low'].values
        support_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if support_slope > 0.001:
            score += 0.25  # Clearly rising
        elif support_slope > 0.0005:
            score += 0.15  # Moderately rising
        else:
            return PatternResult(
                detected=False,
                pattern_name="Ascending Triangle",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=score,
                description="Support line not rising sufficiently",
                action="Not a valid ascending triangle",
                rules=[f"Support slope: {support_slope:.6f}"]
            )
        
        # Volume pattern (should decline)
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.20
        elif abs(volume_trend) < np.std(df['Volume']) * 0.1:
            score += 0.10  # Neutral volume
        
        # Convergence (range should narrow)
        early_range = df.head(10)['High'].max() - df.head(10)['Low'].min()
        late_range = df.tail(10)['High'].max() - df.tail(10)['Low'].min()
        
        if late_range < early_range * 0.7:
            score += 0.15  # Good convergence
        elif late_range < early_range * 0.85:
            score += 0.10  # Some convergence
        
        # Near breakout point
        current_close = df['Close'].iloc[-1]
        resistance_level = np.mean(highs[-5:])
        
        if current_close > resistance_level * 0.98:
            score += 0.10
        
        # Pattern detection threshold
        if score >= 0.70:
            resistance = df['High'].max()
            support = df['Low'].min()
            triangle_height = resistance - support
            
            # Calculate entry, stop, targets
            entry_price = resistance * 1.02
            stop_loss = support * 0.98
            target_1 = resistance + triangle_height  # Height projected
            target_2 = target_1 * 1.15
            target_3 = target_1 * 1.30
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Ascending Triangle",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Bullish continuation pattern showing accumulation at resistance with rising demand.",
                action="BUY on resistance breakout with volume confirmation",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Resistance touches: Count significant tests",
                    f"Entry: Above ₹{resistance:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f}",
                    f"Target: Triangle height projection",
                    f"Volume: Must surge on breakout"
                ],
                metadata={
                    'resistance_slope': resistance_slope,
                    'support_slope': support_slope,
                    'triangle_height': triangle_height,
                    'convergence_ratio': late_range / early_range if early_range > 0 else 1
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Ascending Triangle",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Triangle formation present but below confidence threshold",
            action="Monitor for breakout",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_flat_base(self, lookback: int = 30) -> PatternResult:
        """
        Detect Flat Base - Institutional accumulation pattern.
        
        Pattern Characteristics:
        - Tight consolidation (<12% range)
        - 5-12 week duration
        - Volume contraction
        - Support holding firmly
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 15:
            return PatternResult(
                detected=False,
                pattern_name="Flat Base",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 15 periods required"]
            )
        
        score = 0.0
        
        # Price range analysis
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].mean()
        
        if price_range < 0.10:
            score += 0.30  # Very tight base
        elif price_range < 0.15:
            score += 0.20  # Acceptable base
        else:
            return PatternResult(
                detected=False,
                pattern_name="Flat Base",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description=f"Base too wide: {price_range:.1%} (need <15%)",
                action="Not a flat base",
                rules=[f"Price range: {price_range:.1%}"]
            )
        
        # Time analysis (assuming daily data)
        base_duration = len(df)
        
        if 25 <= base_duration <= 60:  # 5-12 weeks
            score += 0.20
        elif 15 <= base_duration <= 75:
            score += 0.10
        else:
            return PatternResult(
                detected=False,
                pattern_name="Flat Base",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=score,
                description=f"Duration {base_duration} days outside ideal range (25-60)",
                action="Duration not ideal for flat base",
                rules=[f"Duration: {base_duration} days"]
            )
        
        # Volume contraction
        volume_std = df['Volume'].std()
        volume_mean = df['Volume'].mean()
        
        if volume_mean > 0:
            volume_coefficient = volume_std / volume_mean
            
            if volume_coefficient < 0.4:
                score += 0.20  # Very low volatility in volume
            elif volume_coefficient < 0.6:
                score += 0.10
        
        # Support holding
        support_level = df['Low'].min()
        recent_lows = df['Low'].tail(5).values
        
        # Check if support is being tested and holding
        support_tests = sum(1 for low in recent_lows if low < support_level * 1.02)
        support_holds = all(low > support_level * 0.98 for low in recent_lows)
        
        if support_holds and support_tests > 0:
            score += 0.20
        elif support_holds:
            score += 0.10
        
        # Check for previous uptrend (base should form after advance)
        if len(self.data) > lookback * 2:
            prior_data = self.data.iloc[-(lookback * 2):-lookback]
            if len(prior_data) > 10:
                prior_trend = np.polyfit(range(len(prior_data)), prior_data['Close'].values, 1)[0]
                if prior_trend > 0.001:
                    score += 0.10  # Base forming after uptrend
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            base_high = df['High'].max()
            base_low = df['Low'].min()
            
            # Calculate entry, stop, targets
            entry_price = base_high * 1.02
            stop_loss = base_low * 0.97
            target_1 = current_price * 1.20
            target_2 = current_price * 1.35
            target_3 = current_price * 1.50
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Flat Base",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Institutional accumulation pattern showing tight consolidation after advance.",
                action="BUY on volume-fueled breakout from consolidation",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Base range: {price_range:.1%} over {base_duration} days",
                    f"Entry: Above ₹{base_high:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f}",
                    f"Volume: Must expand on breakout",
                    f"Ideal hold: 3-8 weeks"
                ],
                metadata={
                    'price_range': price_range,
                    'base_duration': base_duration,
                    'support_level': support_level,
                    'volume_coefficient': volume_coefficient if volume_mean > 0 else 0
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Flat Base",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Consolidation present but below flat base threshold",
            action="Monitor for tighter consolidation",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_falling_wedge(self, lookback: int = 40) -> PatternResult:
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
            return PatternResult(
                detected=False,
                pattern_name="Falling Wedge",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 30 periods required"]
            )
        
        score = 0.0
        highs = df['High'].values
        lows = df['Low'].values
        
        # Both trendlines declining
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if high_trend < 0 and low_trend < 0:
            score += 0.25
        elif high_trend < 0 or low_trend < 0:
            score += 0.10
        else:
            return PatternResult(
                detected=False,
                pattern_name="Falling Wedge",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Trendlines not both declining",
                action="Not a falling wedge",
                rules=[f"High slope: {high_trend:.6f}, Low slope: {low_trend:.6f}"]
            )
        
        # Converging (range narrowing)
        early_range = df.head(10)['High'].max() - df.head(10)['Low'].min()
        late_range = df.tail(10)['High'].max() - df.tail(10)['Low'].min()
        
        if early_range > 0:
            convergence_ratio = late_range / early_range
            
            if convergence_ratio < 0.6:
                score += 0.25  # Strong convergence
            elif convergence_ratio < 0.8:
                score += 0.15  # Moderate convergence
            else:
                return PatternResult(
                    detected=False,
                    pattern_name="Falling Wedge",
                    signal=PatternSignal.BULLISH,
                    confidence=PatternConfidence.LOW,
                    score=score,
                    description=f"Range not converging (ratio: {convergence_ratio:.2f})",
                    action="Not a valid wedge",
                    rules=[f"Convergence ratio: {convergence_ratio:.2f}"]
                )
        
        # Volume declining
        volume_trend = np.polyfit(range(len(df)), df['Volume'].values, 1)[0]
        if volume_trend < 0:
            score += 0.20
        elif abs(volume_trend) < np.std(df['Volume']) * 0.1:
            score += 0.10
        
        # Near breakout (tight range recently)
        current_range = df['High'].iloc[-1] - df['Low'].iloc[-1]
        if current_range < late_range * 1.1:
            score += 0.10
        
        # Check for oversold conditions (optional)
        if 'RSI' in df.columns:
            recent_rsi = df['RSI'].tail(5).mean()
            if recent_rsi < 35:
                score += 0.10  # Oversold, better reversal potential
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            upper_line = df['High'].max()
            lower_line = df['Low'].min()
            wedge_height = upper_line - lower_line
            
            # Calculate entry, stop, targets
            entry_price = upper_line * 1.01
            stop_loss = lower_line * 0.97
            target_1 = current_price + wedge_height
            target_2 = target_1 * 1.15
            target_3 = target_1 * 1.30
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Falling Wedge",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Bullish reversal pattern showing decreasing selling pressure in downtrend.",
                action="BUY on upside breakout from wedge with volume",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Wedge height: ₹{wedge_height:.2f}",
                    f"Entry: Above ₹{upper_line:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f}",
                    f"Target: Wedge height projected upward",
                    f"Volume: Should expand on breakout"
                ],
                metadata={
                    'high_trend': high_trend,
                    'low_trend': low_trend,
                    'convergence_ratio': convergence_ratio if early_range > 0 else 0,
                    'wedge_height': wedge_height
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Falling Wedge",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Wedge formation present but below confidence threshold",
            action="Monitor for breakout",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_double_bottom(self, lookback: int = 60) -> PatternResult:
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
            return PatternResult(
                detected=False,
                pattern_name="Double Bottom",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 40 periods required"]
            )
        
        score = 0.0
        prices = df['Close'].values
        
        # Find local minima (bottoms)
        minima_indices = []
        for i in range(2, len(prices) - 2):
            if (prices[i] < prices[i-2] and prices[i] < prices[i-1] and 
                prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                minima_indices.append(i)
        
        if len(minima_indices) < 2:
            return PatternResult(
                detected=False,
                pattern_name="Double Bottom",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient local minima",
                action="Not a double bottom formation",
                rules=["Need at least 2 clear bottoms"]
            )
        
        # Take the last two significant minima
        trough1_idx = minima_indices[-2]
        trough2_idx = minima_indices[-1]
        
        # Ensure reasonable spacing
        if trough2_idx - trough1_idx < 10:
            return PatternResult(
                detected=False,
                pattern_name="Double Bottom",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Bottoms too close together",
                action="Not a valid double bottom",
                rules=["Bottoms should be 10+ periods apart"]
            )
        
        trough1_price = prices[trough1_idx]
        trough2_price = prices[trough2_idx]
        
        # Price similarity (within 3%)
        price_diff = abs(trough1_price - trough2_price) / min(trough1_price, trough2_price)
        
        if price_diff < 0.03:
            score += 0.30  # Excellent similarity
        elif price_diff < 0.05:
            score += 0.20  # Good similarity
        else:
            return PatternResult(
                detected=False,
                pattern_name="Double Bottom",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description=f"Bottom prices differ by {price_diff:.1%} (need <5%)",
                action="Not a valid double bottom",
                rules=[f"Price difference: {price_diff:.1%}"]
            )
        
        # Find neckline (peak between bottoms)
        between_prices = prices[trough1_idx:trough2_idx]
        neckline_idx = trough1_idx + np.argmax(between_prices)
        neckline_price = prices[neckline_idx]
        
        # Neckline should be significantly above bottoms
        bottom_avg = (trough1_price + trough2_price) / 2
        neckline_height = (neckline_price - bottom_avg) / bottom_avg
        
        if neckline_height > 0.08:
            score += 0.25  # Good neckline height
        elif neckline_height > 0.05:
            score += 0.15  # Acceptable height
        
        # Volume analysis
        volume1 = df['Volume'].iloc[trough1_idx]
        volume2 = df['Volume'].iloc[trough2_idx]
        
        if volume1 > volume2 * 1.2:
            score += 0.20  # First bottom has higher volume (panic)
        elif volume1 > volume2:
            score += 0.10
        
        # Current position relative to neckline
        current_price = prices[-1]
        if current_price > neckline_price * 0.98:
            score += 0.15  # Near breakout
        elif current_price > neckline_price * 0.95:
            score += 0.10
        
        # Time symmetry (optional)
        left_side = neckline_idx - trough1_idx
        right_side = trough2_idx - neckline_idx
        time_ratio = min(left_side, right_side) / max(left_side, right_side) if max(left_side, right_side) > 0 else 0
        
        if time_ratio > 0.7:
            score += 0.10  # Good time symmetry
        
        # Pattern detection threshold
        if score >= 0.70:
            bottom_price = min(trough1_price, trough2_price)
            pattern_height = neckline_price - bottom_price
            
            # Calculate entry, stop, targets
            entry_price = neckline_price * 1.02
            stop_loss = bottom_price * 0.98
            target_1 = neckline_price + pattern_height
            target_2 = target_1 * 1.15
            target_3 = target_1 * 1.30
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Double Bottom",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="W-shaped reversal pattern showing strong support at a price level.",
                action="BUY on neckline breakout with volume confirmation",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Bottom similarity: {price_diff:.1%} difference",
                    f"Neckline height: {neckline_height:.1%} above bottoms",
                    f"Entry: Above ₹{neckline_price:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f}",
                    f"Target: Pattern height projected upward"
                ],
                metadata={
                    'price_difference': price_diff,
                    'neckline_height': neckline_height,
                    'pattern_height': pattern_height,
                    'time_symmetry': time_ratio,
                    'volume_ratio': volume1 / volume2 if volume2 > 0 else 0
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Double Bottom",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Double bottom formation present but below confidence threshold",
            action="Monitor for neckline breakout",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    # ============================================================================
    # QULLAMAGGIE SWING PATTERNS
    # ============================================================================
    
    def detect_qullamaggie_breakout(self, lookback: int = 25) -> PatternResult:
        """
        Detect Qullamaggie-style Breakout pattern.
        
        Pattern Characteristics:
        - Stair-step higher lows
        - Volume Dry Up (VDU)
        - Above key EMAs (8, 21)
        - Tight consolidation
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 20:
            return PatternResult(
                detected=False,
                pattern_name="Qullamaggie Breakout",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 20 periods required"]
            )
        
        score = 0.0
        
        # Higher lows (stair-step pattern)
        lows = df['Low'].values
        higher_lows = True
        low_streak = 0
        
        for i in range(1, len(lows)):
            if lows[i] > lows[i-1] * 0.995:  # Allow small variations
                low_streak += 1
            else:
                higher_lows = False
                break
        
        if higher_lows and low_streak >= len(lows) * 0.6:
            score += 0.30  # Strong stair-step
        elif low_streak >= len(lows) * 0.4:
            score += 0.15  # Moderate stair-step
        
        # Volume Dry Up
        recent_vol = df['Volume'].tail(5).mean()
        avg_vol = df['Volume'].mean()
        
        if recent_vol < avg_vol * self.volume_dry_up_threshold:
            score += 0.25  # Strong VDU
        elif recent_vol < avg_vol * 0.8:
            score += 0.15  # Moderate VDU
        
        # Above EMAs
        if 'EMA_8' in df.columns and 'EMA_21' in df.columns:
            current_price = df['Close'].iloc[-1]
            ema_8 = df['EMA_8'].iloc[-1]
            ema_21 = df['EMA_21'].iloc[-1]
            
            if current_price > ema_8 and current_price > ema_21:
                score += 0.25
            elif current_price > ema_21:
                score += 0.15
        
        # Tight range (consolidation)
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].mean()
        
        if price_range < 0.12:
            score += 0.20  # Very tight
        elif price_range < 0.18:
            score += 0.10  # Moderately tight
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            consolidation_high = df['High'].max()
            
            # Calculate entry, stop, targets
            entry_price = consolidation_high * 1.01  # ORH entry
            stop_loss = df['Low'].min() * 0.97
            target_1 = current_price * 1.10  # Quick 10% target
            target_2 = current_price * 1.20
            target_3 = current_price * 1.30
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Qullamaggie Breakout",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Stair-step pattern with Volume Dry Up showing accumulation before breakout.",
                action="Enter at Opening Range High (ORH) with volume confirmation",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"VDU: Recent volume {recent_vol/avg_vol:.1f}x average",
                    f"Entry: ORH above ₹{consolidation_high:.2f}",
                    f"Stop: Below ₹{stop_loss:.2f}",
                    f"Hold: 3-5 days, trail with 10 EMA",
                    f"Volume: First 5-min > 3x average"
                ],
                metadata={
                    'vdu_ratio': recent_vol / avg_vol if avg_vol > 0 else 0,
                    'price_range': price_range,
                    'stair_step_streak': low_streak
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Qullamaggie Breakout",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Breakout setup present but below confidence threshold",
            action="Monitor for VDU and tightening",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_episodic_pivot(self, lookback: int = 10) -> PatternResult:
        """
        Detect Episodic Pivot (EP) - Gap and Go pattern.
        
        Pattern Characteristics:
        - Gap up >2%
        - Huge volume spike (>3x)
        - Holds above gap all day
        - Follow-through next day
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 5:
            return PatternResult(
                detected=False,
                pattern_name="Episodic Pivot",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 5 periods required"]
            )
        
        score = 0.0
        gap_found = False
        gap_day_idx = -1
        
        # Look for significant gap
        for i in range(1, len(df)):
            prev_close = df['Close'].iloc[i-1]
            current_open = df['Open'].iloc[i]
            
            if prev_close > 0:
                gap = (current_open - prev_close) / prev_close
                
                if gap > 0.02:  # 2% gap
                    gap_found = True
                    gap_day_idx = i
                    
                    # Volume spike check
                    current_volume = df['Volume'].iloc[i]
                    avg_prev_volume = df['Volume'].iloc[:i].mean()
                    
                    if current_volume > avg_prev_volume * 3:
                        score += 0.40  # Gap with volume spike
                    elif current_volume > avg_prev_volume * 2:
                        score += 0.30
                    
                    # Gap holds (no fill)
                    day_low = df['Low'].iloc[i]
                    if day_low > prev_close:
                        score += 0.30
                    elif day_low > prev_close * 0.995:
                        score += 0.20
                    
                    break
        
        if not gap_found:
            return PatternResult(
                detected=False,
                pattern_name="Episodic Pivot",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="No significant gap found",
                action="Not an episodic pivot",
                rules=["Need >2% gap up"]
            )
        
        # Follow-through after gap
        if gap_day_idx < len(df) - 1:
            next_day = df.iloc[gap_day_idx + 1]
            gap_day = df.iloc[gap_day_idx]
            
            if next_day['Close'] > gap_day['Close']:
                score += 0.20  # Positive follow-through
            elif next_day['Close'] > gap_day['Close'] * 0.99:
                score += 0.10
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            orh = df['High'].iloc[gap_day_idx]  # Opening Range High
            
            # Calculate entry, stop, targets
            entry_price = orh * 1.005  # Slightly above ORH
            stop_loss = df['Low'].iloc[gap_day_idx] * 0.99
            target_1 = current_price * 1.08  # Quick 8% target
            target_2 = current_price * 1.15
            target_3 = current_price * 1.25
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Episodic Pivot",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Earnings/news driven gap with massive volume and continuation.",
                action="Enter at ORH hold, trade momentum for 2-3 days",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    "Entry: ORH (Opening Range High) hold",
                    "Stop: Below gap day low",
                    "Hold: 2-3 day momentum trade",
                    "Volume: First 5-min > 3x average critical",
                    "Must hold above gap all day"
                ],
                metadata={
                    'gap_percent': gap,
                    'volume_spike': current_volume / avg_prev_volume if avg_prev_volume > 0 else 0,
                    'gap_held': day_low > prev_close
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Episodic Pivot",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Episodic pivot detected but below confidence threshold",
            action="Monitor for follow-through",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_parabolic_short(self, lookback: int = 25) -> PatternResult:
        """
        Detect Parabolic Short - Mean reversion setup.
        
        Pattern Characteristics:
        - >30% move in 2-3 weeks
        - Extended from 10 EMA >15%
        - Volume climax
        - First red day
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 15:
            return PatternResult(
                detected=False,
                pattern_name="Parabolic Short",
                signal=PatternSignal.BEARISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 15 periods required"]
            )
        
        score = 0.0
        
        # Strong recent gains
        price_gain = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        if price_gain > 0.35:
            score += 0.30  # Very parabolic
        elif price_gain > 0.25:
            score += 0.20  # Parabolic
        else:
            return PatternResult(
                detected=False,
                pattern_name="Parabolic Short",
                signal=PatternSignal.BEARISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description=f"Price gain {price_gain:.1%} insufficient (need >25%)",
                action="Not parabolic enough",
                rules=[f"Price gain: {price_gain:.1%}"]
            )
        
        # Distance from EMAs
        if 'EMA_8' in df.columns and 'EMA_21' in df.columns:
            current_price = df['Close'].iloc[-1]
            ema_8 = df['EMA_8'].iloc[-1]
            ema_21 = df['EMA_21'].iloc[-1]
            
            deviation_8 = (current_price - ema_8) / ema_8
            deviation_21 = (current_price - ema_21) / ema_21
            
            if deviation_8 > 0.20 or deviation_21 > 0.25:
                score += 0.30  # Very extended
            elif deviation_8 > 0.15 or deviation_21 > 0.20:
                score += 0.20  # Extended
        
        # Volume climax
        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        
        if recent_volume > avg_volume * 2.5:
            score += 0.20  # Volume climax
        elif recent_volume > avg_volume * 1.5:
            score += 0.10
        
        # First red day (reversal signal)
        if len(df) >= 3:
            if df['Close'].iloc[-1] < df['Close'].iloc[-2]:
                score += 0.20  # First red day
            elif df['Close'].iloc[-1] < df['Open'].iloc[-1]:
                score += 0.10  # Red candle today
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            
            # Calculate entry, stop, targets for short
            entry_price = current_price * 0.995  # Slightly below current
            stop_loss = df['High'].max() * 1.01  # Above recent high
            target_1 = df['EMA_8'].iloc[-1] if 'EMA_8' in df.columns else current_price * 0.92
            target_2 = df['EMA_21'].iloc[-1] if 'EMA_21' in df.columns else current_price * 0.85
            target_3 = target_2 * 0.92
            
            risk = stop_loss - entry_price
            risk_reward = (entry_price - target_1) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Parabolic Short",
                signal=PatternSignal.BEARISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Parabolic move extended from EMAs, primed for mean reversion.",
                action="Short on first red day, target EMA reversion",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Extended: {max(deviation_8, deviation_21):.1%} from EMA",
                    f"Entry: First red day close",
                    f"Stop: Above recent high",
                    f"Target: 8/21 EMA reversion",
                    f"Watch for volume exhaustion"
                ],
                metadata={
                    'price_gain': price_gain,
                    'ema_deviation': max(deviation_8, deviation_21) if 'EMA_8' in df.columns else 0,
                    'volume_climax': recent_volume / avg_volume if avg_volume > 0 else 0
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Parabolic Short",
            signal=PatternSignal.BEARISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Parabolic move detected but below short threshold",
            action="Watch for first red day",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_gap_and_go(self, lookback: int = 5) -> PatternResult:
        """
        Detect Gap and Go - Earnings/news momentum.
        
        Pattern Characteristics:
        - Gap >5%
        - Volume >5x average
        - Gap holds (no fill)
        - Continuation
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 3:
            return PatternResult(
                detected=False,
                pattern_name="Gap and Go",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 3 periods required"]
            )
        
        score = 0.0
        
        # Check for gap (comparing yesterday's close to today's open)
        if len(df) >= 2:
            gap = (df['Open'].iloc[1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            
            if gap > 0.05:
                score += 0.40  # Significant gap
                gap_day = df.iloc[1]
                
                # Volume surge
                gap_volume = gap_day['Volume']
                prev_volume = df['Volume'].iloc[0]
                
                if gap_volume > prev_volume * 4:
                    score += 0.30
                elif gap_volume > prev_volume * 2.5:
                    score += 0.20
                
                # Gap holds (no fill)
                if gap_day['Low'] > df['Close'].iloc[0]:
                    score += 0.30  # Gap holds completely
                elif gap_day['Low'] > df['Close'].iloc[0] * 0.995:
                    score += 0.20  # Minor fill
            else:
                return PatternResult(
                    detected=False,
                    pattern_name="Gap and Go",
                    signal=PatternSignal.BULLISH,
                    confidence=PatternConfidence.LOW,
                    score=0.0,
                    description=f"Gap {gap:.1%} insufficient (need >5%)",
                    action="Not a significant gap",
                    rules=[f"Gap size: {gap:.1%}"]
                )
        else:
            return PatternResult(
                detected=False,
                pattern_name="Gap and Go",
                signal=PatternSignal.BULLISH,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Not enough data for gap analysis",
                action="Wait for more data",
                rules=["Need at least 2 days for gap analysis"]
            )
        
        # Pattern detection threshold
        if score >= 0.70:
            current_price = df['Close'].iloc[-1]
            gap_day_high = df['High'].iloc[1]
            
            # Calculate entry, stop, targets
            entry_price = gap_day_high * 1.005  # Above gap day high
            stop_loss = df['Low'].iloc[1] * 0.99  # Below gap day low
            target_1 = current_price * 1.10
            target_2 = current_price * 1.20
            target_3 = current_price * 1.30
            
            risk = entry_price - stop_loss
            risk_reward = (target_1 - entry_price) / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.90 else PatternConfidence.MEDIUM
            
            return PatternResult(
                detected=True,
                pattern_name="Gap and Go",
                signal=PatternSignal.BULLISH,
                confidence=confidence,
                score=min(score, 1.0),
                description="Earnings or news driven gap with massive volume and no fill.",
                action="Enter on gap hold or continuation, ride momentum",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"Gap: {gap:.1%} up",
                    f"Volume: {gap_volume/prev_volume:.1f}x previous",
                    f"Entry: Above gap day high",
                    f"Stop: Below gap day low",
                    f"No gap fill allowed"
                ],
                metadata={
                    'gap_size': gap,
                    'volume_ratio': gap_volume / prev_volume if prev_volume > 0 else 0,
                    'gap_held': gap_day['Low'] > df['Close'].iloc[0]
                }
            )
        
        return PatternResult(
            detected=False,
            pattern_name="Gap and Go",
            signal=PatternSignal.BULLISH,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Gap detected but below confidence threshold",
            action="Monitor for continuation",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    def detect_abcd_pattern(self, lookback: int = 50) -> PatternResult:
        """
        Detect ABCD Harmonic Pattern.
        
        Pattern Characteristics:
        - AB = CD in price and time
        - BC retracement 61.8-78.6% of AB
        - CD extension 127.2-161.8% of BC
        """
        df = self.data.tail(lookback).copy()
        
        if len(df) < 40:
            return PatternResult(
                detected=False,
                pattern_name="ABCD Pattern",
                signal=PatternSignal.REVERSAL,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient data",
                action="Wait for more data",
                rules=["Minimum 40 periods required"]
            )
        
        score = 0.0
        prices = df['Close'].values
        
        # Find significant swing points
        swings = []
        for i in range(2, len(prices)-2):
            # Peak detection
            if (prices[i] > prices[i-2] and prices[i] > prices[i-1] and 
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                swings.append(('peak', i, prices[i]))
            # Trough detection
            elif (prices[i] < prices[i-2] and prices[i] < prices[i-1] and 
                  prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                swings.append(('trough', i, prices[i]))
        
        if len(swings) < 4:
            return PatternResult(
                detected=False,
                pattern_name="ABCD Pattern",
                signal=PatternSignal.REVERSAL,
                confidence=PatternConfidence.LOW,
                score=0.0,
                description="Insufficient swing points",
                action="Not an ABCD pattern",
                rules=["Need at least 4 swing points (A, B, C, D)"]
            )
        
        # Look for ABCD pattern: A (peak), B (trough), C (peak), D (trough) for bearish
        # or A (trough), B (peak), C (trough), D (peak) for bullish
        best_pattern = None
        best_score = 0
        
        for i in range(len(swings)-3):
            type_a, idx_a, price_a = swings[i]
            type_b, idx_b, price_b = swings[i+1]
            type_c, idx_c, price_c = swings[i+2]
            type_d, idx_d, price_d = swings[i+3]
            
            # Check for bullish ABCD: A(trough), B(peak), C(trough), D(peak)
            if (type_a == 'trough' and type_b == 'peak' and 
                type_c == 'trough' and type_d == 'peak'):
                
                AB = price_b - price_a  # Price difference A to B
                BC = price_b - price_c  # Price difference B to C
                CD = price_d - price_c  # Price difference C to D
                
                if AB > 0 and BC > 0 and CD > 0:
                    # Fibonacci ratios
                    bc_ab_ratio = BC / AB
                    cd_bc_ratio = CD / BC
                    
                    pattern_score = 0
                    
                    # BC should retrace 61.8-78.6% of AB
                    if 0.618 <= bc_ab_ratio <= 0.786:
                        pattern_score += 0.35
                    elif 0.55 <= bc_ab_ratio <= 0.85:
                        pattern_score += 0.20
                    
                    # CD should extend 127.2-161.8% of BC
                    if 1.272 <= cd_bc_ratio <= 1.618:
                        pattern_score += 0.35
                    elif 1.15 <= cd_bc_ratio <= 1.75:
                        pattern_score += 0.20
                    
                    # AB and CD should be roughly equal
                    ab_cd_ratio = AB / CD
                    if 0.85 <= ab_cd_ratio <= 1.15:
                        pattern_score += 0.20
                    elif 0.75 <= ab_cd_ratio <= 1.25:
                        pattern_score += 0.10
                    
                    # Time symmetry (optional)
                    ab_time = idx_b - idx_a
                    cd_time = idx_d - idx_c
                    if ab_time > 0 and cd_time > 0:
                        time_ratio = min(ab_time, cd_time) / max(ab_time, cd_time)
                        if time_ratio > 0.7:
                            pattern_score += 0.10
                    
                    if pattern_score > best_score:
                        best_score = pattern_score
                        best_pattern = {
                            'type': 'bullish',
                            'points': [price_a, price_b, price_c, price_d],
                            'indices': [idx_a, idx_b, idx_c, idx_d],
                            'ratios': {
                                'bc_ab': bc_ab_ratio,
                                'cd_bc': cd_bc_ratio,
                                'ab_cd': ab_cd_ratio
                            }
                        }
            
            # Check for bearish ABCD: A(peak), B(trough), C(peak), D(trough)
            elif (type_a == 'peak' and type_b == 'trough' and 
                  type_c == 'peak' and type_d == 'trough'):
                
                AB = price_a - price_b  # Price difference A to B
                BC = price_c - price_b  # Price difference B to C
                CD = price_c - price_d  # Price difference C to D
                
                if AB > 0 and BC > 0 and CD > 0:
                    # Fibonacci ratios
                    bc_ab_ratio = BC / AB
                    cd_bc_ratio = CD / BC
                    
                    pattern_score = 0
                    
                    # BC should retrace 61.8-78.6% of AB
                    if 0.618 <= bc_ab_ratio <= 0.786:
                        pattern_score += 0.35
                    elif 0.55 <= bc_ab_ratio <= 0.85:
                        pattern_score += 0.20
                    
                    # CD should extend 127.2-161.8% of BC
                    if 1.272 <= cd_bc_ratio <= 1.618:
                        pattern_score += 0.35
                    elif 1.15 <= cd_bc_ratio <= 1.75:
                        pattern_score += 0.20
                    
                    # AB and CD should be roughly equal
                    ab_cd_ratio = AB / CD
                    if 0.85 <= ab_cd_ratio <= 1.15:
                        pattern_score += 0.20
                    elif 0.75 <= ab_cd_ratio <= 1.25:
                        pattern_score += 0.10
                    
                    if pattern_score > best_score:
                        best_score = pattern_score
                        best_pattern = {
                            'type': 'bearish',
                            'points': [price_a, price_b, price_c, price_d],
                            'indices': [idx_a, idx_b, idx_c, idx_d],
                            'ratios': {
                                'bc_ab': bc_ab_ratio,
                                'cd_bc': cd_bc_ratio,
                                'ab_cd': ab_cd_ratio
                            }
                        }
        
        score = best_score
        
        # Pattern detection threshold
        if score >= 0.70 and best_pattern:
            signal = PatternSignal.BULLISH if best_pattern['type'] == 'bullish' else PatternSignal.BEARISH
            current_price = df['Close'].iloc[-1]
            d_point = best_pattern['points'][3]
            
            # Calculate entry, stop, targets
            if signal == PatternSignal.BULLISH:
                entry_price = d_point * 1.01  # Slightly above D point
                stop_loss = best_pattern['points'][2] * 0.98  # Below C point
                target_1 = d_point + (best_pattern['points'][1] - best_pattern['points'][0])  # AB projected
                target_2 = target_1 * 1.15
                target_3 = target_1 * 1.30
            else:  # Bearish
                entry_price = d_point * 0.99  # Slightly below D point
                stop_loss = best_pattern['points'][2] * 1.02  # Above C point
                target_1 = d_point - (best_pattern['points'][0] - best_pattern['points'][1])  # AB projected
                target_2 = target_1 * 0.92
                target_3 = target_1 * 0.85
            
            risk = abs(entry_price - stop_loss)
            reward = abs(target_1 - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            confidence = PatternConfidence.HIGH if score > 0.85 else PatternConfidence.MEDIUM
            
            pattern_desc = {
                'bullish': "Bullish harmonic pattern completing at D point for reversal up.",
                'bearish': "Bearish harmonic pattern completing at D point for reversal down."
            }
            
            action_desc = {
                'bullish': "BUY at D point completion with confirmation",
                'bearish': "SHORT at D point completion with confirmation"
            }
            
            return PatternResult(
                detected=True,
                pattern_name="ABCD Pattern",
                signal=signal,
                confidence=confidence,
                score=min(score, 1.0),
                description=pattern_desc[best_pattern['type']],
                action=action_desc[best_pattern['type']],
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                risk_reward=risk_reward,
                rules=[
                    f"BC retracement: {best_pattern['ratios']['bc_ab']:.1%} of AB (ideal: 61.8-78.6%)",
                    f"CD extension: {best_pattern['ratios']['cd_bc']:.1%} of BC (ideal: 127.2-161.8%)",
                    f"AB=CD ratio: {best_pattern['ratios']['ab_cd']:.2f}:1 (ideal: 1:1)",
                    f"Entry: At D point completion",
                    f"Stop: Beyond C point",
                    f"Target: AB=CD projection"
                ],
                metadata=best_pattern
            )
        
        return PatternResult(
            detected=False,
            pattern_name="ABCD Pattern",
            signal=PatternSignal.REVERSAL,
            confidence=PatternConfidence.LOW,
            score=min(score, 1.0),
            description="Harmonic pattern detected but below confidence threshold",
            action="Monitor for D point completion",
            rules=[f"Current score: {score:.2f}/1.0 (need 0.70)"]
        )
    
    # ============================================================================
    # MAIN DETECTION METHODS
    # ============================================================================
    
    def detect_all_zanger_patterns(self) -> List[Dict]:
        """Detect all Dan Zanger patterns."""
        patterns = []
        
        detectors = [
            self.detect_cup_and_handle,
            self.detect_high_tight_flag,
            self.detect_ascending_triangle,
            self.detect_flat_base,
            self.detect_falling_wedge,
            self.detect_double_bottom
        ]
        
        for detector in detectors:
            try:
                result = detector()
                if result.detected:
                    patterns.append(result.to_dict())
                    self.pattern_history.append(result)
            except Exception as e:
                # Log error but continue with other detectors
                if __debug__:
                    print(f"Error in {detector.__name__}: {e}")
                continue
        
        return patterns
    
    def detect_all_swing_patterns(self) -> List[Dict]:
        """Detect all Qullamaggie swing patterns."""
        patterns = []
        
        detectors = [
            self.detect_qullamaggie_breakout,
            self.detect_episodic_pivot,
            self.detect_parabolic_short,
            self.detect_gap_and_go,
            self.detect_abcd_pattern
        ]
        
        for detector in detectors:
            try:
                result = detector()
                if result.detected:
                    patterns.append(result.to_dict())
                    self.pattern_history.append(result)
            except Exception as e:
                # Log error but continue with other detectors
                if __debug__:
                    print(f"Error in {detector.__name__}: {e}")
                continue
        
        return patterns
    
    def detect_all_patterns(self) -> Dict[str, List[Dict]]:
        """Detect all patterns across all categories."""
        zanger_patterns = self.detect_all_zanger_patterns()
        swing_patterns = self.detect_all_swing_patterns()
        
        return {
            'zanger_patterns': zanger_patterns,
            'swing_patterns': swing_patterns,
            'all_patterns': zanger_patterns + swing_patterns,
            'statistics': self.get_detection_statistics()
        }
    
    def get_detection_statistics(self) -> Dict:
        """Get statistics about pattern detections."""
        if not self.pattern_history:
            return {
                'total_detections': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'average_confidence': 0,
                'pattern_frequency': {}
            }
        
        pattern_counts = {}
        bullish_count = 0
        bearish_count = 0
        total_confidence = 0
        
        for pattern in self.pattern_history:
            pattern_name = pattern.pattern_name
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
            
            if pattern.signal == PatternSignal.BULLISH:
                bullish_count += 1
            elif pattern.signal in [PatternSignal.BEARISH, PatternSignal.REVERSAL]:
                bearish_count += 1
            
            # Convert confidence to numeric score
            conf_score = {
                PatternConfidence.LOW: 0.3,
                PatternConfidence.MEDIUM: 0.6,
                PatternConfidence.HIGH: 0.8,
                PatternConfidence.VERY_HIGH: 0.95
            }.get(pattern.confidence, 0.5)
            
            total_confidence += conf_score
        
        return {
            'total_detections': len(self.pattern_history),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'average_confidence': total_confidence / len(self.pattern_history) if self.pattern_history else 0,
            'pattern_frequency': pattern_counts
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_pattern_summary(pattern_dict: Dict) -> str:
    """
    Format pattern details into readable summary.
    
    Args:
        pattern_dict: Pattern dictionary from to_dict()
        
    Returns:
        Formatted string summary
    """
    if not pattern_dict.get('detected', False):
        return "❌ No pattern detected"
    
    summary = f"""
    🎯 **{pattern_dict['pattern']}** - {pattern_dict['signal']}
    
    📊 **Confidence:** {pattern_dict['confidence']} (Score: {pattern_dict['score']:.2f}/1.0)
    
    📝 **Description:** {pattern_dict['description']}
    
    ⚡ **Action:** {pattern_dict['action']}
    
    💰 **Trade Setup:**
    - Entry: {pattern_dict.get('entry_point', 'N/A')}
    - Stop Loss: {pattern_dict.get('stop_loss', 'N/A')}
    - Target 1: {pattern_dict.get('target_1', 'N/A')}
    - Target 2: {pattern_dict.get('target_2', 'N/A')}
    - Risk/Reward: {pattern_dict.get('risk_reward', 'N/A')}
    
    📋 **Rules:**
    """
    
    for rule in pattern_dict.get('rules', []):
        summary += f"\n    • {rule}"
    
    return summary


def get_pattern_statistics(patterns: List[Dict]) -> Dict:
    """
    Calculate statistics across detected patterns.
    
    Args:
        patterns: List of detected patterns
        
    Returns:
        Dictionary with pattern statistics
    """
    if not
