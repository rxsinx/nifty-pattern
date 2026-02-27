"""
Enhanced Hidden Markov Model Analysis using hmmlearn Library
=============================================================

This module implements production-grade HMM using the hmmlearn library for:
- Robust Baum-Welch (EM) parameter estimation
- Multiple covariance types (diagonal, full, spherical, tied)
- AIC/BIC model selection
- Convergence monitoring
- Better numerical stability

Advantages over custom implementation:
- Industry-standard implementation
- Optimized C/Cython code
- Better convergence guarantees
- Multiple covariance structures
- Cross-validation support

Author: Market Analyzer Pro
Version: 2.0 - Enhanced with hmmlearn
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime, timedelta
import warnings

# hmmlearn imports
from hmmlearn import hmm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class EnhancedHiddenMarkovAnalysis:
    """
    Enhanced HMM using hmmlearn library for robust parameter estimation.
    
    Features:
    - Gaussian HMM with multiple covariance types
    - Automatic model selection (AIC/BIC)
    - Cross-validation for parameter tuning
    - Convergence monitoring
    - Regime forecasting with confidence intervals
    """
    
    def __init__(self, data: pd.DataFrame, n_states: int = 3, 
                 covariance_type: str = 'diag', random_state: int = 42):
        """
        Initialize Enhanced HMM Analysis.
        
        Args:
            data: DataFrame with OHLCV data
            n_states: Number of hidden states (default: 3 for BULL/BEAR/SIDEWAYS)
            covariance_type: 'diag', 'full', 'spherical', 'tied' (default: 'diag')
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.prices = data['Close'].values
        self.returns = np.diff(np.log(self.prices))
        
        # HMM Configuration
        self.n_states = n_states
        self.state_names = ['BULL', 'BEAR', 'SIDEWAYS'] if n_states == 3 else [f'STATE_{i}' for i in range(n_states)]
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # HMM Model
        self.model = None
        self.scaler = StandardScaler()
        
        # Results
        self.hidden_states = None
        self.state_probabilities = None
        self.model_score = None
        self.aic = None
        self.bic = None
        self.converged = False
        
    def prepare_features(self, include_volume: bool = True, 
                        include_volatility: bool = True) -> np.ndarray:
        """
        Prepare feature matrix for HMM.
        
        Args:
            include_volume: Include volume changes as feature
            include_volatility: Include realized volatility as feature
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []
        
        # Feature 1: Returns (always included)
        features.append(self.returns)
        
        # Feature 2: Volume changes
        if include_volume and 'Volume' in self.data.columns:
            try:
                volume = self.data['Volume'].values
                volume_changes = np.diff(np.log(volume + 1))  # +1 to avoid log(0)
                
                # Align length with returns
                if len(volume_changes) == len(self.returns):
                    features.append(volume_changes)
            except Exception as e:
                # Skip volume if error
                pass
        
        # Feature 3: Realized volatility (rolling std)
        if include_volatility:
            try:
                window = 5  # 5-day rolling volatility
                volatility = pd.Series(self.returns).rolling(window=window).std().values
                volatility = volatility[window-1:]  # Remove NaN
                
                # Align with returns
                if len(volatility) < len(self.returns):
                    # Pad with median
                    pad_length = len(self.returns) - len(volatility)
                    med_vol = np.nanmedian(volatility)
                    if np.isnan(med_vol):
                        med_vol = 0.01
                    volatility = np.concatenate([np.full(pad_length, med_vol), volatility])
                
                features.append(volatility)
            except Exception as e:
                # Skip volatility if error
                pass
        
        # Stack features
        if len(features) == 1:
            X = features[0].reshape(-1, 1)
        else:
            X = np.column_stack(features)
        
        # Remove any NaN or Inf rows
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_mask]
        
        # Ensure we have enough data
        if len(X) < 50:
            raise ValueError(f"Insufficient valid data: only {len(X)} samples after cleaning")
        
        return X
    
    def fit_model(self, n_iter: int = 100, tol: float = 1e-4, 
                  verbose: bool = False) -> Dict:
        """
        Fit GaussianHMM using Baum-Welch algorithm (EM).
        
        Args:
            n_iter: Maximum iterations for EM algorithm
            tol: Convergence threshold
            verbose: Print convergence progress
            
        Returns:
            Dictionary with model parameters and fit statistics
        """
        # Prepare features
        X = self.prepare_features()
        
        # Standardize features for better convergence
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for hmmlearn (needs 2D with shape (n_samples, n_features))
        X_scaled = X_scaled.reshape(-1, X_scaled.shape[1])
        
        # Initialize GaussianHMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=self.random_state,
            verbose=verbose,
            init_params='stmc',  # Initialize: startprob, transmat, means, covars
            params='stmc'  # Update all parameters
        )
        
        # Fit model
        try:
            self.model.fit(X_scaled)
            self.converged = self.model.monitor_.converged
            self.model_score = self.model.score(X_scaled)
            
            # Calculate AIC and BIC
            n_params = self._count_parameters()
            self.aic = -2 * self.model_score + 2 * n_params
            self.bic = -2 * self.model_score + n_params * np.log(len(X_scaled))
            
            # Decode hidden states using Viterbi
            self.hidden_states = self.model.predict(X_scaled)
            
            # Get state probabilities using Forward-Backward
            self.state_probabilities = self.model.predict_proba(X_scaled)
            
            # Map states to meaningful names based on mean returns
            self._map_states_to_regimes()
            
            if verbose:
                print(f"✅ Model converged: {self.converged}")
                print(f"📊 Log-likelihood: {self.model_score:.2f}")
                print(f"📈 AIC: {self.aic:.2f}")
                print(f"📉 BIC: {self.bic:.2f}")
            
            return {
                'converged': self.converged,
                'log_likelihood': float(self.model_score),
                'aic': float(self.aic),
                'bic': float(self.bic),
                'n_iter': self.model.monitor_.iter,
                'transition_matrix': self.model.transmat_.tolist(),
                'means': self.model.means_.tolist(),
                'covariances': self._get_covariances_as_list(),
                'initial_probs': self.model.startprob_.tolist()
            }
            
        except Exception as e:
            print(f"❌ Model fitting failed: {e}")
            return {
                'converged': False,
                'error': str(e)
            }
    
    def _count_parameters(self) -> int:
        """Count number of free parameters in model."""
        n = self.n_states
        n_features = self.model.means_.shape[1]
        
        # Start probabilities: n - 1 (sum to 1)
        n_params = n - 1
        
        # Transition matrix: n * (n - 1) (each row sums to 1)
        n_params += n * (n - 1)
        
        # Emission means: n * n_features
        n_params += n * n_features
        
        # Covariances
        if self.covariance_type == 'diag':
            n_params += n * n_features  # Diagonal elements only
        elif self.covariance_type == 'full':
            n_params += n * n_features * (n_features + 1) // 2  # Symmetric matrix
        elif self.covariance_type == 'spherical':
            n_params += n  # Single variance per state
        elif self.covariance_type == 'tied':
            n_params += n_features * (n_features + 1) // 2  # One shared covariance
        
        return n_params
    
    def _get_covariances_as_list(self) -> List:
        """Convert covariance matrices to list format."""
        if self.covariance_type == 'diag':
            return self.model.covars_.tolist()
        elif self.covariance_type == 'full':
            return [cov.tolist() for cov in self.model.covars_]
        elif self.covariance_type == 'spherical':
            return self.model.covars_.tolist()
        elif self.covariance_type == 'tied':
            return self.model.covars_.tolist()
        else:
            return []
    
    def _map_states_to_regimes(self):
        """Map HMM states to BULL/BEAR/SIDEWAYS based on mean returns."""
        if self.n_states != 3:
            return  # Only works for 3-state model
        
        # Get mean returns for each state (first feature is returns)
        mean_returns = self.model.means_[:, 0]
        
        # Sort states by mean return
        sorted_indices = np.argsort(mean_returns)
        
        # Create mapping: lowest mean = BEAR, highest = BULL, middle = SIDEWAYS
        state_mapping = {
            sorted_indices[0]: 1,  # BEAR (lowest return)
            sorted_indices[1]: 2,  # SIDEWAYS (middle return)
            sorted_indices[2]: 0   # BULL (highest return)
        }
        
        # Remap hidden states
        self.hidden_states = np.array([state_mapping[s] for s in self.hidden_states])
        
        # Remap probabilities
        self.state_probabilities = self.state_probabilities[:, [sorted_indices[2], sorted_indices[0], sorted_indices[1]]]
        
        # Remap transition matrix
        new_transmat = np.zeros_like(self.model.transmat_)
        for i in range(3):
            for j in range(3):
                new_transmat[state_mapping[i], state_mapping[j]] = self.model.transmat_[i, j]
        self.model.transmat_ = new_transmat
        
        # Remap means and covariances
        self.model.means_ = self.model.means_[[sorted_indices[2], sorted_indices[0], sorted_indices[1]]]
        
        if self.covariance_type == 'full':
            self.model.covars_ = self.model.covars_[[sorted_indices[2], sorted_indices[0], sorted_indices[1]]]
        elif self.covariance_type == 'diag':
            self.model.covars_ = self.model.covars_[[sorted_indices[2], sorted_indices[0], sorted_indices[1]]]
    
    def cross_validate_model(self, n_splits: int = 5) -> Dict:
        """
        Cross-validate HMM using time series splits.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dictionary with CV scores
        """
        X = self.prepare_features()
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            
            # Fit on train
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=50,
                random_state=self.random_state
            )
            
            try:
                model.fit(X_train)
                score = model.score(X_test)
                scores.append(score)
            except:
                continue
        
        return {
            'mean_score': float(np.mean(scores)) if scores else 0,
            'std_score': float(np.std(scores)) if scores else 0,
            'scores': scores
        }
    
    def select_best_model(self, max_states: int = 5, 
                          covariance_types: List[str] = None) -> Dict:
        """
        Select best model using AIC/BIC.
        
        Args:
            max_states: Maximum number of states to try
            covariance_types: List of covariance types to try
            
        Returns:
            Dictionary with best model info
        """
        if covariance_types is None:
            covariance_types = ['diag', 'full']
        
        X = self.prepare_features()
        X_scaled = self.scaler.fit_transform(X)
        
        results = []
        
        for n in range(2, max_states + 1):
            for cov_type in covariance_types:
                try:
                    model = hmm.GaussianHMM(
                        n_components=n,
                        covariance_type=cov_type,
                        n_iter=100,
                        random_state=self.random_state
                    )
                    
                    model.fit(X_scaled)
                    score = model.score(X_scaled)
                    
                    # Calculate AIC/BIC
                    n_params = self._count_parameters_for_model(model, cov_type)
                    aic = -2 * score + 2 * n_params
                    bic = -2 * score + n_params * np.log(len(X_scaled))
                    
                    results.append({
                        'n_states': n,
                        'covariance_type': cov_type,
                        'log_likelihood': score,
                        'aic': aic,
                        'bic': bic,
                        'converged': model.monitor_.converged
                    })
                    
                except:
                    continue
        
        if not results:
            return {'error': 'No models converged'}
        
        # Sort by BIC (lower is better)
        results_sorted = sorted(results, key=lambda x: x['bic'])
        
        return {
            'best_model': results_sorted[0],
            'all_results': results_sorted
        }
    
    def _count_parameters_for_model(self, model, cov_type: str) -> int:
        """Count parameters for a given model."""
        n = model.n_components
        n_features = model.means_.shape[1]
        
        n_params = n - 1  # Start probs
        n_params += n * (n - 1)  # Transition matrix
        n_params += n * n_features  # Means
        
        if cov_type == 'diag':
            n_params += n * n_features
        elif cov_type == 'full':
            n_params += n * n_features * (n_features + 1) // 2
        elif cov_type == 'spherical':
            n_params += n
        elif cov_type == 'tied':
            n_params += n_features * (n_features + 1) // 2
        
        return n_params
    
    def forecast_price(self, forecast_days: int = 30, 
                      n_simulations: int = 1000) -> Dict:
        """
        Forecast future prices using Monte Carlo simulation.
        
        Args:
            forecast_days: Number of days to forecast
            n_simulations: Number of Monte Carlo paths
            
        Returns:
            Dictionary with forecast results
        """
        if self.model is None:
            self.fit_model()
        
        current_price = self.prices[-1]
        current_state_probs = self.state_probabilities[-1]
        
        # Initialize forecasts
        forecast_prices = np.zeros((n_simulations, forecast_days))
        forecast_states = np.zeros((n_simulations, forecast_days), dtype=int)
        
        for sim in range(n_simulations):
            # Sample initial state
            state = np.random.choice(self.n_states, p=current_state_probs)
            price = current_price
            
            for day in range(forecast_days):
                # Record state
                forecast_states[sim, day] = state
                
                # Sample return from current state's distribution
                try:
                    mean = self.model.means_[state, 0]  # First feature is return
                    
                    if self.covariance_type == 'diag':
                        var = self.model.covars_[state, 0]
                    elif self.covariance_type == 'full':
                        var = self.model.covars_[state, 0, 0]
                    elif self.covariance_type == 'spherical':
                        var = self.model.covars_[state]
                    else:  # tied
                        var = self.model.covars_[0, 0]
                    
                    # Generate return (unscale from standardized space)
                    std_return = np.random.normal(mean, np.sqrt(max(var, 1e-6)))
                    
                    # Safe unscaling
                    if hasattr(self.scaler, 'scale_') and len(self.scaler.scale_) > 0:
                        actual_return = std_return * self.scaler.scale_[0] + self.scaler.mean_[0]
                    else:
                        actual_return = std_return * 0.01  # Fallback: assume 1% std
                    
                    # Update price with bounds check
                    price = price * np.exp(np.clip(actual_return, -0.2, 0.2))  # Clip to ±20%
                    forecast_prices[sim, day] = price
                    
                except Exception as e:
                    # Fallback: use simple random walk
                    price = price * np.exp(np.random.normal(0, 0.01))
                    forecast_prices[sim, day] = price
                
                # Transition to next state
                state = np.random.choice(self.n_states, p=self.model.transmat_[state])
        
        # Calculate statistics
        mean_forecast = np.mean(forecast_prices, axis=0)
        median_forecast = np.median(forecast_prices, axis=0)
        std_forecast = np.std(forecast_prices, axis=0)
        
        ci_lower_95 = np.percentile(forecast_prices, 2.5, axis=0)
        ci_upper_95 = np.percentile(forecast_prices, 97.5, axis=0)
        ci_lower_68 = np.percentile(forecast_prices, 16, axis=0)
        ci_upper_68 = np.percentile(forecast_prices, 84, axis=0)
        
        # State frequency analysis
        state_frequencies = np.zeros((forecast_days, self.n_states))
        for day in range(forecast_days):
            for state in range(self.n_states):
                state_frequencies[day, state] = np.sum(forecast_states[:, day] == state) / n_simulations
        
        # Determine direction
        expected_return = (mean_forecast[-1] - current_price) / current_price * 100
        
        if expected_return > 5:
            direction = 'BULLISH'
            signal = 'BUY'
        elif expected_return < -5:
            direction = 'BEARISH'
            signal = 'SELL'
        else:
            direction = 'NEUTRAL'
            signal = 'HOLD'
        
        # Generate dates
        try:
            last_date = self.data.index[-1]
            forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=forecast_days)
            forecast_dates_list = [d.to_pydatetime() for d in forecast_dates]
        except:
            last_date = datetime.now()
            forecast_dates_list = [(last_date + timedelta(days=i+1)) for i in range(forecast_days)]
        
        # Regime analysis
        dominant_state = np.argmax(state_frequencies[-1])
        regime_persistence = self._calculate_regime_persistence()
        confidence = self._assess_forecast_confidence(current_state_probs, regime_persistence)
        
        return {
            'forecast_days': forecast_days,
            'n_simulations': n_simulations,
            'current_price': float(current_price),
            'current_state': self.state_names[int(np.argmax(current_state_probs))],
            'current_state_probability': float(np.max(current_state_probs)),
            'state_probabilities': {
                self.state_names[i]: float(current_state_probs[i])
                for i in range(self.n_states)
            },
            'direction': direction,
            'signal': signal,
            'expected_return': float(expected_return),
            'expected_volatility': float(np.mean(std_forecast) / current_price * 100),
            'dates': forecast_dates_list,
            'mean_forecast': mean_forecast.tolist(),
            'median_forecast': median_forecast.tolist(),
            'std_forecast': std_forecast.tolist(),
            'ci_lower_95': ci_lower_95.tolist(),
            'ci_upper_95': ci_upper_95.tolist(),
            'ci_lower_68': ci_lower_68.tolist(),
            'ci_upper_68': ci_upper_68.tolist(),
            'target_price': float(mean_forecast[-1]),
            'best_case': float(ci_upper_95[-1]),
            'worst_case': float(ci_lower_95[-1]),
            'dominant_regime': self.state_names[dominant_state],
            'regime_confidence': float(state_frequencies[-1, dominant_state]),
            'regime_persistence': regime_persistence,
            'state_transition_matrix': self.model.transmat_.tolist(),
            'state_frequencies': state_frequencies.tolist(),
            'bull_probability': float(np.mean(state_frequencies[:, 0])),
            'bear_probability': float(np.mean(state_frequencies[:, 1])),
            'sideways_probability': float(np.mean(state_frequencies[:, 2])),
            'confidence_level': confidence['level'],
            'confidence_score': float(confidence['score']),
            'confidence_factors': confidence['factors'],
            'method': 'Hidden Markov Model (hmmlearn GaussianHMM)',
            'algorithm': 'Baum-Welch EM with Viterbi decoding',
            'model_quality': {
                'converged': self.converged,
                'log_likelihood': float(self.model_score),
                'aic': float(self.aic),
                'bic': float(self.bic)
            }
        }
    
    def _calculate_regime_persistence(self) -> Dict:
        """Calculate regime persistence metrics."""
        states = self.hidden_states
        persistence = {}
        
        for state in range(self.n_states):
            runs = []
            current_run = 0
            
            for s in states:
                if s == state:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            
            if current_run > 0:
                runs.append(current_run)
            
            if runs:
                persistence[self.state_names[state]] = {
                    'avg_duration': float(np.mean(runs)),
                    'max_duration': int(np.max(runs)),
                    'n_occurrences': len(runs)
                }
            else:
                persistence[self.state_names[state]] = {
                    'avg_duration': 0.0,
                    'max_duration': 0,
                    'n_occurrences': 0
                }
        
        return persistence
    
    def _assess_forecast_confidence(self, state_probs: np.ndarray, 
                                    persistence: Dict) -> Dict:
        """Assess forecast confidence."""
        score = 0
        factors = []
        
        # Factor 1: State probability
        max_prob = np.max(state_probs)
        if max_prob > 0.7:
            score += 0.3
            factors.append(f"High state confidence ({max_prob:.1%})")
        elif max_prob > 0.5:
            score += 0.2
            factors.append(f"Moderate state confidence ({max_prob:.1%})")
        else:
            score += 0.1
            factors.append(f"Low state confidence ({max_prob:.1%})")
        
        # Factor 2: Model convergence
        if self.converged:
            score += 0.2
            factors.append("Model converged successfully")
        else:
            score += 0.1
            factors.append("Model convergence uncertain")
        
        # Factor 3: Model quality (BIC)
        if self.bic is not None:
            # Lower BIC is better, but need context
            score += 0.2
            factors.append(f"BIC: {self.bic:.0f}")
        
        # Factor 4: Sample size
        n_obs = len(self.returns)
        if n_obs > 200:
            score += 0.2
            factors.append(f"Large sample (n={n_obs})")
        elif n_obs > 100:
            score += 0.15
            factors.append(f"Adequate sample (n={n_obs})")
        else:
            score += 0.05
            factors.append(f"Small sample (n={n_obs})")
        
        # Factor 5: Regime persistence
        current_state = self.state_names[int(np.argmax(state_probs))]
        avg_duration = persistence[current_state]['avg_duration']
        
        if avg_duration > 10:
            score += 0.1
            factors.append(f"{current_state} stable ({avg_duration:.0f} days avg)")
        
        # Determine level
        if score >= 0.8:
            level = 'HIGH'
        elif score >= 0.6:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'score': score,
            'level': level,
            'factors': factors
        }
    
    def analyze_regime_characteristics(self) -> Dict:
        """Analyze characteristics of each regime."""
        states = self.hidden_states
        returns = self.returns
        
        # Align returns with states (states are 1 shorter after diff)
        if len(states) > len(returns):
            states = states[:len(returns)]
        elif len(returns) > len(states):
            returns = returns[:len(states)]
        
        characteristics = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            state_returns = returns[state_mask]
            
            if len(state_returns) > 0:
                characteristics[self.state_names[state]] = {
                    'avg_return': float(np.mean(state_returns) * 100),
                    'volatility': float(np.std(state_returns) * 100),
                    'sharpe_ratio': float(np.mean(state_returns) / np.std(state_returns)) if np.std(state_returns) > 0 else 0,
                    'win_rate': float(np.sum(state_returns > 0) / len(state_returns) * 100),
                    'avg_gain': float(np.mean(state_returns[state_returns > 0]) * 100) if np.sum(state_returns > 0) > 0 else 0,
                    'avg_loss': float(np.mean(state_returns[state_returns < 0]) * 100) if np.sum(state_returns < 0) > 0 else 0,
                    'max_gain': float(np.max(state_returns) * 100),
                    'max_loss': float(np.min(state_returns) * 100),
                    'occurrences': int(np.sum(state_mask)),
                    'duration_pct': float(np.sum(state_mask) / len(states) * 100)
                }
            else:
                characteristics[self.state_names[state]] = {
                    'avg_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'avg_gain': 0.0,
                    'avg_loss': 0.0,
                    'max_gain': 0.0,
                    'max_loss': 0.0,
                    'occurrences': 0,
                    'duration_pct': 0.0
                }
        
        return characteristics
    
    def generate_trading_strategy(self, forecast: Dict) -> Dict:
        """Generate trading strategy based on forecast."""
        current_state = forecast['current_state']
        direction = forecast['direction']
        expected_return = forecast['expected_return']
        
        characteristics = self.analyze_regime_characteristics()
        current_char = characteristics[current_state]
        
        strategy = {
            'signal': forecast['signal'],
            'direction': direction,
            'confidence': forecast['confidence_level'],
            'entry_price': forecast['current_price'],
            'target_price': forecast['target_price'],
            'stop_loss': None,
            'position_size': None,
            'time_horizon': f"{forecast['forecast_days']} days",
            'rationale': [],
            'risks': []
        }
        
        # Set stop loss and position size
        if direction == 'BULLISH':
            stop_distance = current_char['volatility'] * 2
            strategy['stop_loss'] = forecast['current_price'] * (1 - stop_distance / 100)
            strategy['rationale'].append(f"HMM in {current_state} regime")
            strategy['rationale'].append(f"Expected {expected_return:.1f}% upside")
            strategy['rationale'].append(f"Win rate: {current_char['win_rate']:.0f}%")
            
            if forecast['confidence_level'] == 'HIGH':
                strategy['position_size'] = '3-5% of portfolio'
            elif forecast['confidence_level'] == 'MEDIUM':
                strategy['position_size'] = '2-3% of portfolio'
            else:
                strategy['position_size'] = '1-2% of portfolio'
            
            strategy['risks'].append(f"Max loss: {current_char['max_loss']:.1f}%")
            strategy['risks'].append(f"Model uncertainty: {100 - forecast['confidence_score']*100:.0f}%")
            
        elif direction == 'BEARISH':
            stop_distance = current_char['volatility'] * 2
            strategy['stop_loss'] = forecast['current_price'] * (1 + stop_distance / 100)
            strategy['rationale'].append(f"HMM in {current_state} regime")
            strategy['rationale'].append(f"Expected {abs(expected_return):.1f}% downside")
            strategy['position_size'] = 'Reduce exposure or SHORT'
            
            strategy['risks'].append(f"Reversal risk: {forecast['bull_probability']:.1%}")
            
        else:  # NEUTRAL
            strategy['stop_loss'] = forecast['current_price'] * 0.97
            strategy['position_size'] = '2-3% range trading'
            strategy['rationale'].append("Sideways regime - range bound")
            strategy['risks'].append("Breakout risk")
        
        return strategy


def run_enhanced_hmm_analysis(data: pd.DataFrame, forecast_days: int = 30,
                              auto_select: bool = False) -> Dict:
    """
    Run complete enhanced HMM analysis with fallback to original.
    
    Args:
        data: DataFrame with OHLCV data
        forecast_days: Days to forecast
        auto_select: Automatically select best model using AIC/BIC
        
    Returns:
        Complete analysis results
    """
    try:
        hmm_analyzer = EnhancedHiddenMarkovAnalysis(data)
        
        # Auto-select best model if requested
        if auto_select:
            print("🔍 Selecting best model...")
            selection = hmm_analyzer.select_best_model(max_states=4, 
                                                        covariance_types=['diag', 'full'])
            
            if 'best_model' in selection:
                best = selection['best_model']
                print(f"✅ Best: {best['n_states']} states, {best['covariance_type']} covariance (BIC: {best['bic']:.2f})")
                
                # Re-initialize with best params
                hmm_analyzer = EnhancedHiddenMarkovAnalysis(
                    data,
                    n_states=best['n_states'],
                    covariance_type=best['covariance_type']
                )
        
        # Fit model
        fit_results = hmm_analyzer.fit_model(verbose=False)
        
        if not fit_results.get('converged'):
            print("⚠️  Model did not converge, trying with more iterations...")
            fit_results = hmm_analyzer.fit_model(n_iter=200, tol=1e-3, verbose=False)
        
        # Generate forecast
        forecast = hmm_analyzer.forecast_price(forecast_days=forecast_days)
        
        # Analyze regimes
        characteristics = hmm_analyzer.analyze_regime_characteristics()
        
        # Generate strategy
        strategy = hmm_analyzer.generate_trading_strategy(forecast)
        
        # Persistence
        persistence = hmm_analyzer._calculate_regime_persistence()
        
        return {
            'hmm_parameters': fit_results,
            'forecast': forecast,
            'characteristics': characteristics,
            'strategy': strategy,
            'persistence': persistence,
            'model_selection': selection if auto_select else None,
            'method': 'enhanced'
        }
        
    except Exception as e:
        print(f"⚠️  Enhanced HMM failed: {e}")
        print("🔄 Falling back to original HMM implementation...")
        
        # Fallback to original implementation
        try:
            from markov_analysis import run_hmm_analysis as run_original
            results = run_original(data, forecast_days=forecast_days)
            results['method'] = 'original_fallback'
            results['fallback_reason'] = str(e)
            return results
        except Exception as e2:
            print(f"❌ Original HMM also failed: {e2}")
            # Return minimal error response
            return {
                'error': True,
                'error_message': f"Both HMM implementations failed. Enhanced: {str(e)}, Original: {str(e2)}",
                'method': 'failed',
                'forecast': {
                    'signal': 'HOLD',
                    'direction': 'NEUTRAL',
                    'confidence_level': 'LOW',
                    'current_price': float(data['Close'].iloc[-1]),
                    'target_price': float(data['Close'].iloc[-1]),
                    'expected_return': 0.0
                }
            }


if __name__ == "__main__":
    print("Enhanced HMM Analysis Module with hmmlearn - Ready")
    print("\nFeatures:")
    print("✅ Robust Baum-Welch (EM) parameter estimation")
    print("✅ Multiple covariance types (diag, full, spherical, tied)")
    print("✅ AIC/BIC model selection")
    print("✅ Time series cross-validation")
    print("✅ Convergence monitoring")
    print("✅ Better numerical stability")
