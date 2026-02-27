#!/usr/bin/env python3
"""
Quick Test Script: Original vs Enhanced HMM
============================================

Run this to test both implementations on sample data.

Usage:
    python test_hmm_comparison.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_comparison():
    """Test both HMM implementations."""
    
    print("🧪 HMM Implementation Test")
    print("=" * 80)
    print()
    
    # Step 1: Create sample data
    print("📊 Step 1: Creating sample data...")
    
    try:
        import yfinance as yf
        import pandas as pd
        
        # Fetch NIFTY data
        data = yf.Ticker("^NSEI").history(period="1y")
        
        if data.empty:
            print("❌ Failed to fetch data. Using synthetic data instead.")
            # Create synthetic data
            import numpy as np
            dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
            np.random.seed(42)
            prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02))
            
            data = pd.DataFrame({
                'Open': prices,
                'High': prices * 1.01,
                'Low': prices * 0.99,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, 252)
            }, index=dates)
        
        print(f"✅ Data loaded: {len(data)} days")
        print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Test Original Implementation
    print("🔵 Step 2: Testing ORIGINAL Implementation...")
    print("-" * 80)
    
    try:
        from markov_analysis import run_hmm_analysis
        import time
        
        start = time.time()
        original_results = run_hmm_analysis(data, forecast_days=30)
        original_time = time.time() - start
        
        orig_forecast = original_results['forecast']
        
        print(f"✅ Success in {original_time:.2f}s")
        print(f"   Current: ₹{orig_forecast['current_price']:.2f}")
        print(f"   Target:  ₹{orig_forecast['target_price']:.2f} ({orig_forecast['expected_return']:+.2f}%)")
        print(f"   Signal:  {orig_forecast['signal']}")
        print(f"   Confidence: {orig_forecast['confidence_level']}")
        print()
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print()
        original_results = None
        original_time = 0
    
    # Step 3: Test Enhanced Implementation
    print("🟢 Step 3: Testing ENHANCED Implementation (hmmlearn)...")
    print("-" * 80)
    
    try:
        from markov_analysis_hmmlearn import run_enhanced_hmm_analysis
        import time
        
        start = time.time()
        enhanced_results = run_enhanced_hmm_analysis(data, forecast_days=30, auto_select=False)
        enhanced_time = time.time() - start
        
        enh_forecast = enhanced_results['forecast']
        enh_params = enhanced_results['hmm_parameters']
        
        print(f"✅ Success in {enhanced_time:.2f}s")
        print(f"   Converged: {enh_params['converged']}")
        print(f"   Log-likelihood: {enh_params['log_likelihood']:.2f}")
        print(f"   AIC: {enh_params['aic']:.2f}")
        print(f"   BIC: {enh_params['bic']:.2f}")
        print(f"   Current: ₹{enh_forecast['current_price']:.2f}")
        print(f"   Target:  ₹{enh_forecast['target_price']:.2f} ({enh_forecast['expected_return']:+.2f}%)")
        print(f"   Signal:  {enh_forecast['signal']}")
        print(f"   Confidence: {enh_forecast['confidence_level']}")
        print()
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"   Please install: pip install hmmlearn scikit-learn")
        print()
        enhanced_results = None
        enhanced_time = 0
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print()
        enhanced_results = None
        enhanced_time = 0
    
    # Step 4: Compare Results
    if original_results and enhanced_results:
        print("📊 Step 4: Comparison")
        print("=" * 80)
        print()
        
        orig_target = original_results['forecast']['target_price']
        enh_target = enhanced_results['forecast']['target_price']
        
        print(f"⏱️  Execution Time:")
        print(f"   Original: {original_time:.3f}s")
        print(f"   Enhanced: {enhanced_time:.3f}s")
        print(f"   Difference: {(enhanced_time - original_time):.3f}s")
        print()
        
        print(f"🎯 Target Price:")
        print(f"   Original: ₹{orig_target:.2f}")
        print(f"   Enhanced: ₹{enh_target:.2f}")
        print(f"   Difference: ₹{abs(orig_target - enh_target):.2f} ({abs(orig_target - enh_target)/orig_target*100:.1f}%)")
        print()
        
        orig_signal = original_results['forecast']['signal']
        enh_signal = enhanced_results['forecast']['signal']
        
        print(f"🚦 Trading Signal:")
        print(f"   Original: {orig_signal}")
        print(f"   Enhanced: {enh_signal}")
        print(f"   Agreement: {'✅ Yes' if orig_signal == enh_signal else '❌ No'}")
        print()
        
        print(f"📈 Enhanced-Only Metrics:")
        print(f"   Converged: {enhanced_results['hmm_parameters']['converged']}")
        print(f"   AIC: {enhanced_results['hmm_parameters']['aic']:.2f} (lower is better)")
        print(f"   BIC: {enhanced_results['hmm_parameters']['bic']:.2f} (lower is better)")
        print()
        
        print("💡 Recommendation:")
        if enhanced_results['hmm_parameters']['converged']:
            print("   ✅ Use ENHANCED version - better statistical properties")
        else:
            print("   ⚠️  Enhanced did not converge - consider ORIGINAL or adjust parameters")
        
    print()
    print("=" * 80)
    print("✅ Test Complete!")
    print()
    print("📝 Next Steps:")
    print("   1. Review hmm_comparison_guide.py for full migration guide")
    print("   2. Update app.py imports to use enhanced version")
    print("   3. Test with your actual stock data")
    print("=" * 80)


if __name__ == "__main__":
    test_comparison()
