"""
Quantitative Analysis Module for Advanced Statistical Trading
==============================================================

This module contains advanced quantitative methods including:
- Fractal Analysis (Hurst Exponent, Fractal Dimension)
- Maximum Likelihood Estimation (MLE) and Bayesian Estimation
- Volatility Modelling (GARCH, EWMA, Parkinson, Garman-Klass)

Author: Market Analyzer Pro
Version: 1.0 - Quantitative Finance Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class FractalAnalysis:
    """
    Fractal Analysis for detecting market patterns and trends.
    
    Methods:
    - Hurst Exponent: Measure of long-term memory and trend strength
    - Fractal Dimension: Market complexity and randomness
    - Rescaled Range Analysis (R/S): Persistence vs anti-persistence
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Fractal Analysis with price data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self.prices = data['Close'].values
    
    def calculate_hurst_exponent(self, lags: Optional[List[int]] = None) -> Dict:
        """
        Calculate Hurst Exponent using Rescaled Range (R/S) Analysis.
        
        Interpretation:
        - H = 0.5: Random walk (Brownian motion)
        - H > 0.5: Trending market (persistent)
        - H < 0.5: Mean reverting (anti-persistent)
        
        Args:
            lags: List of lag periods for analysis
            
        Returns:
            Dictionary with Hurst exponent and interpretation
        """
        if lags is None:
            # Use logarithmically spaced lags
            lags = [2, 4, 8, 16, 32, 64, 128]
        
        # Filter lags that are smaller than data length
        lags = [lag for lag in lags if lag < len(self.prices) // 2]
        
        if len(lags) < 3:
            return {
                'hurst_exponent': 0.5,
                'interpretation': 'INSUFFICIENT_DATA',
                'confidence': 'LOW',
                'market_behavior': 'RANDOM',
                'lags_used': lags
            }
        
        tau = []
        rs_values = []
        
        for lag in lags:
            # Divide price series into subseries of length lag
            n_subseries = len(self.prices) // lag
            
            if n_subseries == 0:
                continue
            
            rs_lag = []
            
            for i in range(n_subseries):
                subseries = self.prices[i * lag:(i + 1) * lag]
                
                if len(subseries) < 2:
                    continue
                
                # Mean of subseries
                mean = np.mean(subseries)
                
                # Mean-adjusted series
                Y = subseries - mean
                
                # Cumulative deviate
                Z = np.cumsum(Y)
                
                # Range
                R = np.max(Z) - np.min(Z)
                
                # Standard deviation
                S = np.std(subseries, ddof=1)
                
                # R/S ratio (avoid division by zero)
                if S > 0:
                    rs_lag.append(R / S)
            
            if rs_lag:
                tau.append(lag)
                rs_values.append(np.mean(rs_lag))
        
        if len(tau) < 3:
            return {
                'hurst_exponent': 0.5,
                'interpretation': 'INSUFFICIENT_DATA',
                'confidence': 'LOW',
                'market_behavior': 'RANDOM',
                'lags_used': tau
            }
        
        # Linear regression: log(R/S) = H * log(tau) + c
        log_tau = np.log(tau)
        log_rs = np.log(rs_values)
        
        # Perform linear regression
        coeffs = np.polyfit(log_tau, log_rs, 1)
        hurst = coeffs[0]
        
        # Interpretation
        if hurst > 0.55:
            interpretation = 'TRENDING'
            market_behavior = 'PERSISTENT'
            confidence = 'HIGH' if hurst > 0.65 else 'MEDIUM'
            recommendation = 'Use trend-following strategies'
        elif hurst < 0.45:
            interpretation = 'MEAN_REVERTING'
            market_behavior = 'ANTI_PERSISTENT'
            confidence = 'HIGH' if hurst < 0.35 else 'MEDIUM'
            recommendation = 'Use mean reversion strategies'
        else:
            interpretation = 'RANDOM_WALK'
            market_behavior = 'RANDOM'
            confidence = 'MEDIUM'
            recommendation = 'Market is random - use risk management'
        
        return {
            'hurst_exponent': float(hurst),
            'interpretation': interpretation,
            'market_behavior': market_behavior,
            'confidence': confidence,
            'recommendation': recommendation,
            'lags_used': tau,
            'rs_values': rs_values,
            'r_squared': self._calculate_r_squared(log_tau, log_rs, coeffs)
        }
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Calculate R-squared for linear fit."""
        y_pred = coeffs[0] * x + coeffs[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return float(r_squared)
    
    def calculate_fractal_dimension(self, max_lag: int = 20) -> Dict:
        """
        Calculate Fractal Dimension using box-counting method.
        
        Interpretation:
        - FD = 1.0: Perfectly smooth (strong trend)
        - FD = 1.5: Random walk
        - FD = 2.0: Highly irregular (choppy market)
        
        Args:
            max_lag: Maximum lag for calculation
            
        Returns:
            Dictionary with fractal dimension and interpretation
        """
        returns = np.diff(np.log(self.prices))
        
        if len(returns) < max_lag:
            max_lag = len(returns) // 2
        
        lags = range(1, max_lag + 1)
        variances = []
        
        for lag in lags:
            # Calculate variance at different scales
            lagged_returns = returns[::lag]
            if len(lagged_returns) > 1:
                variance = np.var(lagged_returns)
                variances.append(variance)
            else:
                break
        
        if len(variances) < 3:
            return {
                'fractal_dimension': 1.5,
                'interpretation': 'INSUFFICIENT_DATA',
                'market_state': 'UNKNOWN',
                'confidence': 'LOW'
            }
        
        # Log-log regression
        valid_lags = list(range(1, len(variances) + 1))
        log_lags = np.log(valid_lags)
        log_vars = np.log(variances)
        
        # Fit: log(var) = 2*H*log(lag) + c
        coeffs = np.polyfit(log_lags, log_vars, 1)
        hurst_alt = coeffs[0] / 2
        
        # Fractal Dimension: FD = 2 - H
        fractal_dim = 2 - hurst_alt
        
        # Interpretation
        if fractal_dim < 1.3:
            interpretation = 'STRONG_TREND'
            market_state = 'TRENDING'
            confidence = 'HIGH'
            recommendation = 'Strong directional bias - trend following'
        elif fractal_dim < 1.7:
            interpretation = 'WEAK_TREND'
            market_state = 'SLIGHTLY_TRENDING'
            confidence = 'MEDIUM'
            recommendation = 'Mild trend - use breakout strategies'
        else:
            interpretation = 'CHOPPY_MARKET'
            market_state = 'RANGE_BOUND'
            confidence = 'HIGH' if fractal_dim > 1.8 else 'MEDIUM'
            recommendation = 'Choppy/ranging - use mean reversion'
        
        return {
            'fractal_dimension': float(fractal_dim),
            'interpretation': interpretation,
            'market_state': market_state,
            'confidence': confidence,
            'recommendation': recommendation,
            'hurst_estimate': float(hurst_alt)
        }
    
    def detect_fractals(self, periods: int = 5) -> pd.DataFrame:
        """
        Detect fractal highs and lows (Williams Fractals).
        
        A fractal high: Middle high is higher than 2 highs on each side
        A fractal low: Middle low is lower than 2 lows on each side
        
        Args:
            periods: Number of bars on each side (default 5 = 2 bars each side)
            
        Returns:
            DataFrame with fractal signals
        """
        df = self.data.copy()
        n = periods // 2
        
        df['Fractal_High'] = False
        df['Fractal_Low'] = False
        df['Fractal_High_Price'] = np.nan
        df['Fractal_Low_Price'] = np.nan
        
        for i in range(n, len(df) - n):
            # Fractal High
            if df['High'].iloc[i] == max(df['High'].iloc[i - n:i + n + 1]):
                df.loc[df.index[i], 'Fractal_High'] = True
                df.loc[df.index[i], 'Fractal_High_Price'] = df['High'].iloc[i]
            
            # Fractal Low
            if df['Low'].iloc[i] == min(df['Low'].iloc[i - n:i + n + 1]):
                df.loc[df.index[i], 'Fractal_Low'] = True
                df.loc[df.index[i], 'Fractal_Low_Price'] = df['Low'].iloc[i]
        
        return df[['Fractal_High', 'Fractal_Low', 'Fractal_High_Price', 'Fractal_Low_Price']]


class StatisticalEstimation:
    """
    Maximum Likelihood Estimation (MLE) and Bayesian Estimation.
    
    Methods:
    - MLE for return distribution parameters
    - Bayesian estimation with prior distributions
    - Model selection using AIC/BIC
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Statistical Estimation with price data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self.returns = np.diff(np.log(data['Close'].values))
    
    def mle_normal_distribution(self) -> Dict:
        """
        Maximum Likelihood Estimation for Normal Distribution.
        
        Returns:
            Dictionary with MLE parameters and statistics
        """
        # MLE estimates
        mu_mle = np.mean(self.returns)
        sigma_mle = np.std(self.returns, ddof=1)
        
        # Log-likelihood
        n = len(self.returns)
        log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(sigma_mle**2) - \
                        np.sum((self.returns - mu_mle)**2) / (2 * sigma_mle**2)
        
        # AIC and BIC
        k = 2  # Number of parameters
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Goodness of fit (Kolmogorov-Smirnov test)
        ks_stat, ks_pvalue = stats.kstest(self.returns, 'norm', args=(mu_mle, sigma_mle))
        
        # Annualized statistics (assuming daily data)
        annual_return = mu_mle * 252
        annual_volatility = sigma_mle * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'distribution': 'NORMAL',
            'mu_mle': float(mu_mle),
            'sigma_mle': float(sigma_mle),
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'normal_fit': 'GOOD' if ks_pvalue > 0.05 else 'POOR',
            'annual_return': float(annual_return * 100),  # Percentage
            'annual_volatility': float(annual_volatility * 100),  # Percentage
            'sharpe_ratio': float(sharpe_ratio),
            'n_observations': n
        }
    
    def mle_students_t_distribution(self) -> Dict:
        """
        Maximum Likelihood Estimation for Student's t-Distribution.
        Better for fat-tailed returns.
        
        Returns:
            Dictionary with MLE parameters
        """
        # MLE for Student's t
        params = stats.t.fit(self.returns)
        df_t, loc, scale = params
        
        # Log-likelihood
        log_likelihood = np.sum(stats.t.logpdf(self.returns, df_t, loc, scale))
        
        # AIC and BIC
        k = 3  # Number of parameters
        n = len(self.returns)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Goodness of fit
        ks_stat, ks_pvalue = stats.kstest(self.returns, 't', args=params)
        
        return {
            'distribution': 'STUDENTS_T',
            'degrees_of_freedom': float(df_t),
            'location': float(loc),
            'scale': float(scale),
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            't_fit': 'GOOD' if ks_pvalue > 0.05 else 'POOR',
            'tail_heaviness': 'HEAVY' if df_t < 5 else 'MODERATE' if df_t < 10 else 'LIGHT',
            'n_observations': n
        }
    
    def bayesian_estimation(self, prior_mu: float = 0.0, prior_sigma: float = 0.02,
                          prior_strength: float = 10) -> Dict:
        """
        Bayesian Estimation with Normal-Gamma conjugate prior.
        
        Args:
            prior_mu: Prior belief about mean return
            prior_sigma: Prior belief about volatility
            prior_strength: Strength of prior belief (in equivalent observations)
            
        Returns:
            Dictionary with posterior parameters
        """
        n = len(self.returns)
        sample_mean = np.mean(self.returns)
        sample_var = np.var(self.returns, ddof=1)
        
        # Posterior parameters (Normal-Gamma conjugate)
        # Posterior mean is weighted average of prior and sample
        posterior_mu = (prior_strength * prior_mu + n * sample_mean) / (prior_strength + n)
        
        # Posterior variance (accounts for uncertainty)
        posterior_variance = (prior_strength * prior_sigma**2 + (n - 1) * sample_var + 
                            prior_strength * n * (sample_mean - prior_mu)**2 / (prior_strength + n)) / \
                           (prior_strength + n)
        posterior_sigma = np.sqrt(posterior_variance)
        
        # Credible intervals (95%)
        ci_multiplier = 1.96
        ci_lower = posterior_mu - ci_multiplier * posterior_sigma
        ci_upper = posterior_mu + ci_multiplier * posterior_sigma
        
        # Posterior predictive distribution
        predictive_sigma = np.sqrt(posterior_variance * (1 + 1/(prior_strength + n)))
        
        # Probability of positive return
        prob_positive = 1 - stats.norm.cdf(0, posterior_mu, predictive_sigma)
        
        # Annualized statistics
        annual_return = posterior_mu * 252
        annual_volatility = posterior_sigma * np.sqrt(252)
        
        return {
            'method': 'BAYESIAN',
            'posterior_mu': float(posterior_mu),
            'posterior_sigma': float(posterior_sigma),
            'credible_interval_95': (float(ci_lower), float(ci_upper)),
            'predictive_sigma': float(predictive_sigma),
            'prob_positive_return': float(prob_positive * 100),  # Percentage
            'annual_return_estimate': float(annual_return * 100),
            'annual_volatility_estimate': float(annual_volatility * 100),
            'prior_influence': float(prior_strength / (prior_strength + n) * 100),
            'n_observations': n
        }
    
    def model_comparison(self) -> Dict:
        """
        Compare Normal vs Student's t distribution using AIC/BIC.
        
        Returns:
            Dictionary with model comparison results
        """
        normal_results = self.mle_normal_distribution()
        t_results = self.mle_students_t_distribution()
        
        # Lower AIC/BIC is better
        if t_results['aic'] < normal_results['aic']:
            best_model = 'STUDENTS_T'
            aic_improvement = normal_results['aic'] - t_results['aic']
        else:
            best_model = 'NORMAL'
            aic_improvement = t_results['aic'] - normal_results['aic']
        
        if t_results['bic'] < normal_results['bic']:
            bic_best_model = 'STUDENTS_T'
            bic_improvement = normal_results['bic'] - t_results['bic']
        else:
            bic_best_model = 'NORMAL'
            bic_improvement = t_results['bic'] - normal_results['bic']
        
        return {
            'best_model_aic': best_model,
            'best_model_bic': bic_best_model,
            'aic_improvement': float(aic_improvement),
            'bic_improvement': float(bic_improvement),
            'normal_aic': float(normal_results['aic']),
            'students_t_aic': float(t_results['aic']),
            'normal_bic': float(normal_results['bic']),
            'students_t_bic': float(t_results['bic']),
            'recommendation': best_model if best_model == bic_best_model else 'MIXED'
        }


class VolatilityModelling:
    """
    Advanced Volatility Modelling techniques.
    
    Methods:
    - GARCH(1,1): Generalized AutoRegressive Conditional Heteroskedasticity
    - EWMA: Exponentially Weighted Moving Average
    - Parkinson: High-Low volatility estimator
    - Garman-Klass: OHLC volatility estimator
    - Yang-Zhang: Most efficient OHLC estimator
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Volatility Modelling with price data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self.returns = np.diff(np.log(data['Close'].values))
    
    def simple_volatility(self, window: int = 20) -> pd.Series:
        """
        Simple historical volatility (standard deviation of returns).
        
        Args:
            window: Rolling window size
            
        Returns:
            Series of historical volatility
        """
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        volatility = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def ewma_volatility(self, lambda_param: float = 0.94) -> pd.Series:
        """
        Exponentially Weighted Moving Average (EWMA) volatility.
        RiskMetrics approach (lambda = 0.94 for daily data).
        
        Args:
            lambda_param: Decay factor (0.94 is standard for daily)
            
        Returns:
            Series of EWMA volatility
        """
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
        
        # Initialize
        ewma_var = np.zeros(len(log_returns))
        ewma_var[0] = log_returns.iloc[0] ** 2
        
        # Recursive calculation
        for t in range(1, len(log_returns)):
            ewma_var[t] = lambda_param * ewma_var[t-1] + (1 - lambda_param) * log_returns.iloc[t] ** 2
        
        # Convert to volatility (annualized)
        ewma_vol = np.sqrt(ewma_var * 252)
        
        return pd.Series(ewma_vol, index=log_returns.index, name='EWMA_Volatility')
    
    def parkinson_volatility(self, window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator using High-Low range.
        More efficient than close-to-close when no gaps.
        
        Formula: σ² = (1/(4*ln(2))) * E[(ln(H/L))²]
        
        Args:
            window: Rolling window size
            
        Returns:
            Series of Parkinson volatility
        """
        df = self.data.copy()
        
        # High-Low ratio
        hl = np.log(df['High'] / df['Low'])
        
        # Parkinson variance
        parkinson_var = (1 / (4 * np.log(2))) * (hl ** 2).rolling(window=window).mean()
        
        # Convert to volatility (annualized)
        parkinson_vol = np.sqrt(parkinson_var * 252)
        
        return parkinson_vol
    
    def garman_klass_volatility(self, window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator using OHLC.
        More efficient than Parkinson, accounts for opening jumps.
        
        Args:
            window: Rolling window size
            
        Returns:
            Series of Garman-Klass volatility
        """
        df = self.data.copy()
        
        # Log ratios
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        
        # Garman-Klass variance
        gk_var = 0.5 * (log_hl ** 2).rolling(window=window).mean() - \
                 (2 * np.log(2) - 1) * (log_co ** 2).rolling(window=window).mean()
        
        # Convert to volatility (annualized)
        gk_vol = np.sqrt(gk_var * 252)
        
        return gk_vol
    
    def yang_zhang_volatility(self, window: int = 20) -> pd.Series:
        """
        Yang-Zhang volatility estimator - most efficient OHLC estimator.
        Accounts for overnight jumps and intraday volatility.
        
        Args:
            window: Rolling window size
            
        Returns:
            Series of Yang-Zhang volatility
        """
        df = self.data.copy()
        
        # Log ratios
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_co = np.log(df['Close'] / df['Open'])
        
        log_oc = np.log(df['Open'] / df['Close'].shift(1))
        log_cc = np.log(df['Close'] / df['Close'].shift(1))
        
        # Overnight volatility
        overnight_vol = (log_oc ** 2).rolling(window=window).mean()
        
        # Open-close volatility
        open_close_vol = (log_co ** 2).rolling(window=window).mean()
        
        # Rogers-Satchell volatility component
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window=window).mean()
        
        # Yang-Zhang variance
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_vol + k * open_close_vol + (1 - k) * rs
        
        # Convert to volatility (annualized)
        yz_vol = np.sqrt(yz_var * 252)
        
        return yz_vol
    
    def garch_volatility(self, p: int = 1, q: int = 1) -> Dict:
        """
        GARCH(p,q) volatility model estimation.
        GARCH(1,1) is most common: σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)
        
        Args:
            p: GARCH lag order
            q: ARCH lag order
            
        Returns:
            Dictionary with GARCH parameters and forecasts
        """
        returns = self.returns
        
        # Initial variance estimate
        var_init = np.var(returns)
        
        # GARCH(1,1) optimization
        def garch_likelihood(params):
            omega, alpha, beta = params
            
            # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            # Initialize variance
            var = np.zeros(len(returns))
            var[0] = var_init
            
            # GARCH recursion
            for t in range(1, len(returns)):
                var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
            
            # Log-likelihood (negative for minimization)
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) + returns**2 / var)
            
            return -log_lik
        
        # Initial guess
        initial_params = [var_init * 0.01, 0.05, 0.90]
        
        # Bounds
        bounds = [(1e-6, None), (0, 1), (0, 1)]
        
        # Optimize
        try:
            result = minimize(garch_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                omega, alpha, beta = result.x
                
                # Calculate fitted variances
                var_fitted = np.zeros(len(returns))
                var_fitted[0] = var_init
                
                for t in range(1, len(returns)):
                    var_fitted[t] = omega + alpha * returns[t-1]**2 + beta * var_fitted[t-1]
                
                # One-step ahead forecast
                var_forecast = omega + alpha * returns[-1]**2 + beta * var_fitted[-1]
                vol_forecast = np.sqrt(var_forecast * 252) * 100  # Annualized %
                
                # Long-run variance
                long_run_var = omega / (1 - alpha - beta)
                long_run_vol = np.sqrt(long_run_var * 252) * 100  # Annualized %
                
                # Persistence
                persistence = alpha + beta
                
                return {
                    'model': f'GARCH({p},{q})',
                    'omega': float(omega),
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'persistence': float(persistence),
                    'half_life': float(np.log(0.5) / np.log(persistence)) if persistence < 1 else np.inf,
                    'forecast_volatility': float(vol_forecast),
                    'long_run_volatility': float(long_run_vol),
                    'unconditional_volatility': float(np.sqrt(var_init * 252) * 100),
                    'log_likelihood': float(-result.fun),
                    'aic': float(2 * 3 - 2 * (-result.fun)),
                    'convergence': 'SUCCESS',
                    'fitted_variances': var_fitted.tolist()
                }
            else:
                return {
                    'model': f'GARCH({p},{q})',
                    'convergence': 'FAILED',
                    'message': 'Optimization did not converge'
                }
        
        except Exception as e:
            return {
                'model': f'GARCH({p},{q})',
                'convergence': 'ERROR',
                'message': str(e)
            }
    
    def volatility_comparison(self, window: int = 20) -> pd.DataFrame:
        """
        Compare different volatility estimators.
        
        Args:
            window: Window size for rolling calculations
            
        Returns:
            DataFrame with multiple volatility measures
        """
        df = pd.DataFrame()
        
        df['Simple_Vol'] = self.simple_volatility(window)
        df['EWMA_Vol'] = self.ewma_volatility()
        df['Parkinson_Vol'] = self.parkinson_volatility(window)
        df['GarmanKlass_Vol'] = self.garman_klass_volatility(window)
        df['YangZhang_Vol'] = self.yang_zhang_volatility(window)
        
        return df
    
    def volatility_regime_detection(self, window: int = 20, threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect volatility regimes (Low/Medium/High).
        
        Args:
            window: Window for volatility calculation
            threshold: Multiplier for regime detection
            
        Returns:
            DataFrame with regime indicators
        """
        vol = self.simple_volatility(window)
        vol_median = vol.median()
        vol_std = vol.std()
        
        regimes = pd.DataFrame(index=self.data.index)
        regimes['Volatility'] = vol
        
        # Define regimes
        regimes['Regime'] = 'MEDIUM'
        regimes.loc[vol < vol_median - threshold * vol_std, 'Regime'] = 'LOW'
        regimes.loc[vol > vol_median + threshold * vol_std, 'Regime'] = 'HIGH'
        
        # Regime changes
        regimes['Regime_Change'] = regimes['Regime'] != regimes['Regime'].shift(1)
        
        return regimes


# Utility functions for integration
def run_full_quantitative_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete quantitative analysis suite.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary with all analysis results
    """
    results = {}
    
    # Fractal Analysis
    fractal = FractalAnalysis(data)
    results['fractal'] = {
        'hurst_exponent': fractal.calculate_hurst_exponent(),
        'fractal_dimension': fractal.calculate_fractal_dimension(),
        'williams_fractals': fractal.detect_fractals()
    }
    
    # Statistical Estimation
    stats_est = StatisticalEstimation(data)
    results['statistical_estimation'] = {
        'mle_normal': stats_est.mle_normal_distribution(),
        'mle_students_t': stats_est.mle_students_t_distribution(),
        'bayesian': stats_est.bayesian_estimation(),
        'model_comparison': stats_est.model_comparison()
    }
    
    # Volatility Modelling
    vol_model = VolatilityModelling(data)
    results['volatility'] = {
        'garch': vol_model.garch_volatility(),
        'comparison': vol_model.volatility_comparison(),
        'regimes': vol_model.volatility_regime_detection()
    }
    
    return results


# Example usage
if __name__ == "__main__":
    print("Quantitative Analysis Module - Ready for import")
    print("\nAvailable classes:")
    print("- FractalAnalysis: Hurst Exponent, Fractal Dimension, Williams Fractals")
    print("- StatisticalEstimation: MLE (Normal/Student-t), Bayesian Estimation")
    print("- VolatilityModelling: GARCH, EWMA, Parkinson, Garman-Klass, Yang-Zhang")
    print("\nAvailable function:")
    print("- run_full_quantitative_analysis(): Complete analysis suite")
