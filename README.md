# Pattern Detector Module 📈

Complete pattern detection library for technical analysis of stock charts.

## 🎯 Overview

This module implements professional-grade chart pattern detection algorithms used by master traders:
- **Dan Zanger Patterns**: Cup and Handle, High Tight Flag, Ascending Triangle, etc.
- **Qullamaggie Patterns**: Breakout, Episodic Pivot, Parabolic Short, etc.
- **Entry/Exit Points**: Precise price levels for every pattern
- **Risk Management**: Stop loss and profit targets included

## 📦 Installation

```python
# Simply import the module
from pattern_detector import PatternDetector, format_pattern_summary, get_pattern_statistics
```

## 🚀 Quick Start

```python
import pandas as pd
from pattern_detector import PatternDetector

# Your OHLCV data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# Create detector
detector = PatternDetector(df)

# Detect all patterns
results = detector.detect_all_patterns()

# Get Dan Zanger patterns
zanger_patterns = results['zanger_patterns']

# Get Qullamaggie patterns
swing_patterns = results['swing_patterns']

# Print results
for pattern in results['all_patterns']:
    print(f"Found: {pattern['pattern']}")
    print(f"Signal: {pattern['signal']}")
    print(f"Entry: {pattern['entry_point']}")
    print(f"Stop: {pattern['stop_loss']}")
    print(f"Target: {pattern['target_1']}")
```

## 📚 Pattern Library

### Dan Zanger Patterns

#### 1. Cup and Handle
**Description**: Most powerful bull market pattern  
**Entry**: Breakout above handle with 3x+ volume  
**Success Rate**: 65-75%  
**Example**:
```python
pattern = detector.detect_cup_and_handle()
if pattern['detected']:
    print(f"Entry at {pattern['entry_point']}")
    print(f"Stop at {pattern['stop_loss']}")
```

#### 2. High Tight Flag
**Description**: Rare explosive continuation pattern  
**Entry**: Breakout with massive volume (5x+)  
**Success Rate**: 70-80%  

#### 3. Ascending Triangle
**Description**: Bullish continuation, higher lows  
**Entry**: Resistance breakout with volume  
**Success Rate**: 60-70%  

#### 4. Flat Base
**Description**: Institutional accumulation  
**Entry**: Pivot point breakout  
**Success Rate**: 55-65%  

#### 5. Falling Wedge
**Description**: Bullish reversal pattern  
**Entry**: Upside breakout  
**Success Rate**: 60-70%  

#### 6. Double Bottom
**Description**: W-shaped reversal  
**Entry**: Neckline breakout  
**Success Rate**: 65-75%  

### Qullamaggie Swing Patterns

#### 1. Breakout (High Tight Flag)
**Description**: Stair-step with VDU  
**Entry**: ORH (Opening Range High)  
**Success Rate**: 70-80%  

#### 2. Episodic Pivot
**Description**: Gap and Go momentum  
**Entry**: At ORH, hold 2-3 days  
**Success Rate**: 65-75%  

#### 3. Parabolic Short
**Description**: Mean reversion setup  
**Entry**: First red day after climax  
**Success Rate**: 60-70%  

#### 4. Gap and Go
**Description**: Earnings/news gap continuation  
**Entry**: On gap hold or continuation  
**Success Rate**: 70-80%  

#### 5. ABCD Pattern
**Description**: Harmonic Fibonacci pattern  
**Entry**: D point completion  
**Success Rate**: 55-65%  

## 🔧 API Reference

### PatternDetector Class

```python
class PatternDetector:
    def __init__(self, data: pd.DataFrame):
        """Initialize with OHLCV DataFrame"""
        
    def detect_cup_and_handle(self, lookback: int = 100) -> Dict:
        """Detect Cup and Handle pattern"""
        
    def detect_high_tight_flag(self, lookback: int = 30) -> Dict:
        """Detect High Tight Flag pattern"""
        
    def detect_ascending_triangle(self, lookback: int = 30) -> Dict:
        """Detect Ascending Triangle pattern"""
        
    def detect_flat_base(self, lookback: int = 20) -> Dict:
        """Detect Flat Base pattern"""
        
    def detect_falling_wedge(self, lookback: int = 30) -> Dict:
        """Detect Falling Wedge pattern"""
        
    def detect_double_bottom(self, lookback: int = 40) -> Dict:
        """Detect Double Bottom pattern"""
        
    def detect_qullamaggie_breakout(self, lookback: int = 20) -> Dict:
        """Detect Qullamaggie Breakout pattern"""
        
    def detect_episodic_pivot(self, lookback: int = 10) -> Dict:
        """Detect Episodic Pivot pattern"""
        
    def detect_parabolic_short(self, lookback: int = 20) -> Dict:
        """Detect Parabolic Short pattern"""
        
    def detect_gap_and_go(self, lookback: int = 5) -> Dict:
        """Detect Gap and Go pattern"""
        
    def detect_abcd_pattern(self, lookback: int = 40) -> Dict:
        """Detect ABCD Harmonic pattern"""
        
    def detect_all_zanger_patterns(self) -> List[Dict]:
        """Detect all Dan Zanger patterns"""
        
    def detect_all_swing_patterns(self) -> List[Dict]:
        """Detect all Qullamaggie patterns"""
        
    def detect_all_patterns(self) -> Dict[str, List[Dict]]:
        """Detect all patterns across all categories"""
```

### Utility Functions

```python
def format_pattern_summary(pattern: Dict) -> str:
    """Format pattern into readable summary"""
    
def get_pattern_statistics(patterns: List[Dict]) -> Dict:
    """Calculate statistics across patterns"""
```

## 📊 Pattern Output Format

Each detected pattern returns:

```python
{
    'detected': True,
    'pattern': 'Cup and Handle',
    'signal': 'BULLISH',  # or 'BEARISH', 'NEUTRAL', 'REVERSAL'
    'confidence': 'HIGH',  # or 'MEDIUM', 'LOW'
    'score': 0.85,  # 0.0 to 1.0
    'description': 'Pattern description...',
    'entry_point': '₹2,450.50 (Breakout above handle)',
    'stop_loss': '₹2,320.00 (Below handle low)',
    'target_1': '₹2,680.00 (15% gain)',
    'target_2': '₹2,890.00 (Cup depth projected)',
    'action': 'BUY on breakout with >3x volume',
    'rules': [
        'Minimum 7-8 weeks cup formation',
        'Handle 1-4 weeks',
        'Entry: Above ₹2,450.50',
        'Stop: Below ₹2,320.00'
    ]
}
```

## 💡 Usage Examples

### Example 1: Basic Pattern Detection

```python
from pattern_detector import PatternDetector
import yfinance as yf

# Get stock data
ticker = yf.Ticker("RELIANCE.NS")
df = ticker.history(period="1y")

# Add technical indicators (if needed)
df['EMA_8'] = df['Close'].ewm(span=8).mean()
df['EMA_21'] = df['Close'].ewm(span=21).mean()

# Detect patterns
detector = PatternDetector(df)
patterns = detector.detect_all_patterns()

# Show results
for pattern in patterns['all_patterns']:
    print(f"\n{pattern['pattern']}")
    print(f"Signal: {pattern['signal']}")
    print(f"Confidence: {pattern['confidence']}")
    print(f"Entry: {pattern['entry_point']}")
```

### Example 2: Filter by Confidence

```python
# Get only HIGH confidence patterns
high_conf_patterns = [
    p for p in patterns['all_patterns'] 
    if p['confidence'] == 'HIGH'
]

print(f"Found {len(high_conf_patterns)} high confidence patterns")
```

### Example 3: Get Statistics

```python
from pattern_detector import get_pattern_statistics

stats = get_pattern_statistics(patterns['all_patterns'])

print(f"Total patterns: {stats['total_patterns']}")
print(f"Bullish: {stats['bullish_count']}")
print(f"Bearish: {stats['bearish_count']}")
print(f"Average score: {stats['avg_score']:.2f}")
```

### Example 4: Pattern Screening

```python
# Screen multiple stocks
stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

for symbol in stocks:
    df = yf.Ticker(symbol).history(period="6mo")
    detector = PatternDetector(df)
    results = detector.detect_all_patterns()
    
    if results['all_patterns']:
        print(f"\n{symbol}: {len(results['all_patterns'])} patterns")
        for p in results['all_patterns']:
            print(f"  - {p['pattern']} ({p['signal']})")
```

## 🎓 Trading Rules

### Dan Zanger's Rules
1. **Volume Confirmation**: Every breakout needs 3x+ volume
2. **8% Stop Loss**: Hard stop on all positions
3. **Focus on Leaders**: Only trade liquid, high-volume stocks
4. **Patience**: Wait for perfect setup

### Qullamaggie's Rules
1. **1% Risk Rule**: Never risk more than 1% per trade
2. **Market Leaders**: Only trade stocks in leading sectors
3. **ORH Entry**: Opening Range High for momentum trades
4. **Quick Profits**: Take partial profits quickly, trail winners

## ⚠️ Risk Disclaimer

This module is for **educational purposes only**. Pattern detection does not guarantee trading success. Always:
- Use proper risk management
- Never risk more than 1-2% per trade
- Backtest strategies before live trading
- Consider market conditions
- Consult a financial advisor

## 🔄 Integration with Streamlit

```python
import streamlit as st
from pattern_detector import PatternDetector, get_pattern_statistics

# In your Streamlit app
analyzer = YourAnalyzer(symbol, period)
detector = PatternDetector(analyzer.data)

# Detect patterns
patterns = detector.detect_all_patterns()

# Display in Streamlit
st.subheader("Pattern Detection")

if patterns['all_patterns']:
    # Show statistics
    stats = get_pattern_statistics(patterns['all_patterns'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patterns", stats['total_patterns'])
    col2.metric("Bullish", stats['bullish_count'])
    col3.metric("Bearish", stats['bearish_count'])
    
    # Show patterns
    for pattern in patterns['all_patterns']:
        with st.expander(f"{pattern['pattern']} - {pattern['signal']}"):
            st.write(f"**Confidence**: {pattern['confidence']}")
            st.write(f"**Entry**: {pattern['entry_point']}")
            st.write(f"**Stop**: {pattern['stop_loss']}")
            st.write(f"**Target 1**: {pattern['target_1']}")
            st.write(f"**Target 2**: {pattern['target_2']}")
```

## 📈 Performance

- **Patterns Detected**: 11 unique patterns
- **Detection Speed**: <100ms for 100-day data
- **Accuracy**: Based on master trader strategies
- **Memory**: ~5MB for pattern detection

## 🤝 Contributing

To add new patterns:
1. Create detection method in `PatternDetector` class
2. Follow existing pattern format
3. Return standardized dictionary
4. Add to `detect_all_patterns()` method
5. Document in README

## 📄 License

MIT License - Free to use in personal and commercial projects

## 🙏 Credits

Pattern logic based on strategies from:
- **Dan Zanger**: Chart pattern master, world record holder
- **Kristjan Kullamägi (Qullamaggie)**: Swing trading expert
- **William O'Neil**: CANSLIM methodology

For questions or issues:
- Create GitHub issue
- Check documentation
- Review examples

---

**Version**: 2.0  
**Last Updated**: 2026-02-08  
**Compatibility**: Python 3.7+, Pandas 1.0+, NumPy 1.18+
