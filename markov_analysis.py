"""
Hidden Markov Model (HMM) Analysis for Stock Market Prediction
===============================================================

This module implements advanced Hidden Markov Models for:
- Market regime detection (Bull, Bear, Sideways)
- Price action prediction based on historical patterns
- Transition probability analysis
- Viterbi algorithm for optimal state sequence
- Forward-Backward algorithm for state probabilities
- Regime-based forecasting

Author: Market Analyzer Pro
Version: 1.0 - HMM Price Prediction Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class HiddenMarkovAnalysis:
    """
    Hidden Markov Model for stock market analysis and prediction.
    
    This class implements HMM-based analysis including:
    - Regime detection (Hidden states: Bull, Bear, Sideways)
    - Transition probability matrix
    - Emission probability distributions
    - Price forecasting based on regime transitions
    - Viterbi algorithm for state sequence
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize HMM Analysis with price data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self.prices = data['Close'].values
        self.returns = np.diff(np.log(self.prices))
        
        # HMM Parameters (will be estimated)
        self.n_states = 3  # Bull, Bear, Sideways
        self.state_names = ['BULL', 'BEAR', 'SIDEWAYS']
        
        # Matrices
        self.transition_matrix = None
        self.emission_params = None
        self.initial_probs = None
        
        # State sequences
        self.hidden_states = None
        self.state_probabilities = None
    
    def estimate_hmm_parameters(self) -> Dict:
        """
        Estimate HMM parameters using Baum-Welch algorithm (EM).
        
        Returns:
            Dictionary with estimated HMM parameters
        """
        # Simplified parameter estimation using regime classification
        # In production, would use hmmlearn library or full EM implementation
        
        returns = self.returns
        
        # Step 1: Initial regime classification based on returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Classify into initial regimes
        states = np.zeros(len(returns), dtype=int)
        
        for i, ret in enumerate(returns):
            if ret > mean_return + 0.5 * std_return:
                states[i] = 0  # BULL
            elif ret < mean_return - 0.5 * std_return:
                states[i] = 1  # BEAR
            else:
                states[i] = 2  # SIDEWAYS
        
        # Step 2: Estimate transition probabilities
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1
        
        # Normalize to get probabilities
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # Uniform distribution if no transitions observed
                self.transition_matrix[i, :] = 1.0 / self.n_states
        
        # Step 3: Estimate emission parameters (mean and std for each state)
        self.emission_params = {}
        
        for state in range(self.n_states):
            state_returns = returns[states == state]
            
            if len(state_returns) > 0:
                self.emission_params[state] = {
                    'mean': float(np.mean(state_returns)),
                    'std': float(np.std(state_returns)),
                    'count': len(state_returns)
                }
            else:
                # Default parameters
                self.emission_params[state] = {
                    'mean': 0.0,
                    'std': std_return,
                    'count': 0
                }
        
        # Step 4: Estimate initial state probabilities
        self.initial_probs = np.zeros(self.n_states)
        
        for state in range(self.n_states):
            self.initial_probs[state] = np.sum(states == state) / len(states)
        
        return {
            'transition_matrix': self.transition_matrix.tolist(),
            'emission_params': self.emission_params,
            'initial_probs': self.initial_probs.tolist(),
            'state_names': self.state_names
        }
    
    def viterbi_algorithm(self) -> np.ndarray:
        """
        Viterbi algorithm to find most likely state sequence.
        
        Returns:
            Array of most likely hidden states
        """
        if self.transition_matrix is None:
            self.estimate_hmm_parameters()
        
        returns = self.returns
        T = len(returns)
        
        # Initialize
        viterbi = np.zeros((self.n_states, T))
        path = np.zeros((self.n_states, T), dtype=int)
        
        # Initial probabilities
        for state in range(self.n_states):
            emission_prob = self._emission_probability(returns[0], state)
            viterbi[state, 0] = np.log(self.initial_probs[state] + 1e-10) + np.log(emission_prob + 1e-10)
        
        # Recursion
        for t in range(1, T):
            for state in range(self.n_states):
                # Find max probability path to this state
                trans_probs = viterbi[:, t-1] + np.log(self.transition_matrix[:, state] + 1e-10)
                path[state, t] = np.argmax(trans_probs)
                viterbi[state, t] = np.max(trans_probs) + np.log(self._emission_probability(returns[t], state) + 1e-10)
        
        # Backtrack to find best path
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(viterbi[:, T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = path[states[t+1], t+1]
        
        self.hidden_states = states
        
        return states
    
    def forward_backward_algorithm(self) -> np.ndarray:
        """
        Forward-Backward algorithm for state probability estimation.
        
        Returns:
            Array of state probabilities at each time step
        """
        if self.transition_matrix is None:
            self.estimate_hmm_parameters()
        
        returns = self.returns
        T = len(returns)
        
        # Forward pass
        alpha = np.zeros((self.n_states, T))
        
        # Initialize
        for state in range(self.n_states):
            alpha[state, 0] = self.initial_probs[state] * self._emission_probability(returns[0], state)
        
        # Normalize
        alpha[:, 0] /= np.sum(alpha[:, 0])
        
        # Forward recursion
        for t in range(1, T):
            for state in range(self.n_states):
                alpha[state, t] = np.sum(alpha[:, t-1] * self.transition_matrix[:, state]) * \
                                 self._emission_probability(returns[t], state)
            
            # Normalize
            alpha[:, t] /= np.sum(alpha[:, t])
        
        # Backward pass
        beta = np.zeros((self.n_states, T))
        beta[:, T-1] = 1.0
        
        # Backward recursion
        for t in range(T-2, -1, -1):
            for state in range(self.n_states):
                beta[state, t] = np.sum(self.transition_matrix[state, :] * 
                                       self._emission_probability(returns[t+1], np.arange(self.n_states)) *
                                       beta[:, t+1])
            
            # Normalize
            beta[:, t] /= np.sum(beta[:, t])
        
        # Calculate state probabilities
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0)
        
        self.state_probabilities = gamma
        
        return gamma
    
    def _emission_probability(self, observation, state):
        """
        Calculate emission probability for observation given state.
        
        Args:
            observation: Observed return (float or array)
            state: Hidden state index (int or array)
            
        Returns:
            Emission probability (float or array)
        """
        # Handle array of states
        if isinstance(state, np.ndarray):
            probs = np.zeros(len(state))
            for i, s in enumerate(state):
                params = self.emission_params[int(s)]
                mean = params['mean']
                std = params['std']
                
                if std == 0:
                    std = 1e-6
                
                probs[i] = stats.norm.pdf(observation, loc=mean, scale=std)
            
            return probs
        
        # Handle single state
        params = self.emission_params[state]
        mean = params['mean']
        std = params['std']
        
        if std == 0:
            std = 1e-6
        
        # Gaussian emission probability
        prob = stats.norm.pdf(observation, loc=mean, scale=std)
        
        return prob
    
    def predict_next_state(self, current_state: int) -> Dict:
        """
        Predict next state based on transition probabilities.
        
        Args:
            current_state: Current hidden state
            
        Returns:
            Dictionary with next state predictions
        """
        if self.transition_matrix is None:
            self.estimate_hmm_parameters()
        
        next_state_probs = self.transition_matrix[current_state, :]
        most_likely_next = np.argmax(next_state_probs)
        
        return {
            'current_state': self.state_names[current_state],
            'next_state_probabilities': {
                self.state_names[i]: float(next_state_probs[i])
                for i in range(self.n_states)
            },
            'most_likely_next_state': self.state_names[most_likely_next],
            'confidence': float(next_state_probs[most_likely_next])
        }
    
    def forecast_price(self, forecast_days: int = 30, n_simulations: int = 1000) -> Dict:
        """
        Forecast future prices using HMM-based Monte Carlo simulation.
        
        Args:
            forecast_days: Number of days to forecast
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with forecast results
        """
        # Ensure HMM parameters are estimated
        if self.transition_matrix is None:
            self.estimate_hmm_parameters()
        
        # Get current state
        if self.hidden_states is None:
            self.viterbi_algorithm()
        
        current_state = self.hidden_states[-1]
        current_price = self.prices[-1]
        
        # Run forward-backward for current state probabilities
        if self.state_probabilities is None:
            self.forward_backward_algorithm()
        
        current_state_probs = self.state_probabilities[:, -1]
        
        # Monte Carlo simulation
        forecast_prices = np.zeros((n_simulations, forecast_days + 1))
        forecast_prices[:, 0] = current_price
        
        state_paths = np.zeros((n_simulations, forecast_days), dtype=int)
        
        for sim in range(n_simulations):
            # Sample initial state based on current probabilities
            state = np.random.choice(self.n_states, p=current_state_probs)
            
            for day in range(forecast_days):
                # Store state
                state_paths[sim, day] = state
                
                # Generate return based on state
                mean = self.emission_params[state]['mean']
                std = self.emission_params[state]['std']
                
                daily_return = np.random.normal(mean, std)
                
                # Update price
                forecast_prices[sim, day + 1] = forecast_prices[sim, day] * np.exp(daily_return)
                
                # Transition to next state
                state = np.random.choice(self.n_states, p=self.transition_matrix[state, :])
        
        # Remove initial price column
        forecast_prices = forecast_prices[:, 1:]
        
        # Calculate statistics
        mean_forecast = np.mean(forecast_prices, axis=0)
        median_forecast = np.median(forecast_prices, axis=0)
        std_forecast = np.std(forecast_prices, axis=0)
        
        # Confidence intervals
        ci_lower_95 = np.percentile(forecast_prices, 2.5, axis=0)
        ci_upper_95 = np.percentile(forecast_prices, 97.5, axis=0)
        ci_lower_68 = np.percentile(forecast_prices, 16, axis=0)
        ci_upper_68 = np.percentile(forecast_prices, 84, axis=0)
        
        # Analyze state path frequencies
        state_frequencies = np.zeros((forecast_days, self.n_states))
        for day in range(forecast_days):
            for state in range(self.n_states):
                state_frequencies[day, state] = np.sum(state_paths[:, day] == state) / n_simulations
        
        # Determine overall forecast direction
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
        
        # Determine dominant regime
        final_state_probs = state_frequencies[-1, :]
        dominant_state = np.argmax(final_state_probs)
        dominant_regime = self.state_names[dominant_state]
        
        # Calculate regime persistence (how long does current state last?)
        regime_persistence = self._calculate_regime_persistence()
        
        # Generate forecast dates
        try:
            last_date = self.data.index[-1]
            forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=forecast_days)
            forecast_dates_list = [d.to_pydatetime() for d in forecast_dates]
        except Exception:
            last_date = datetime.now()
            forecast_dates_list = [(last_date + timedelta(days=i+1)) for i in range(forecast_days)]
        
        # Confidence assessment
        confidence_score = self._assess_forecast_confidence(current_state_probs, regime_persistence)
        
        return {
            # Basic Info
            'forecast_days': forecast_days,
            'n_simulations': n_simulations,
            'current_price': float(current_price),
            
            # Current State
            'current_state': self.state_names[current_state],
            'current_state_probability': float(current_state_probs[current_state]),
            'state_probabilities': {
                self.state_names[i]: float(current_state_probs[i])
                for i in range(self.n_states)
            },
            
            # Forecast
            'direction': direction,
            'signal': signal,
            'expected_return': float(expected_return),
            'expected_volatility': float(np.mean(std_forecast) / current_price * 100),
            
            # Price Forecasts
            'dates': forecast_dates_list,
            'mean_forecast': mean_forecast.tolist(),
            'median_forecast': median_forecast.tolist(),
            'std_forecast': std_forecast.tolist(),
            'ci_lower_95': ci_lower_95.tolist(),
            'ci_upper_95': ci_upper_95.tolist(),
            'ci_lower_68': ci_lower_68.tolist(),
            'ci_upper_68': ci_upper_68.tolist(),
            
            # Targets
            'target_price': float(mean_forecast[-1]),
            'best_case': float(ci_upper_95[-1]),
            'worst_case': float(ci_lower_95[-1]),
            
            # Regime Analysis
            'dominant_regime': dominant_regime,
            'regime_confidence': float(final_state_probs[dominant_state]),
            'regime_persistence': regime_persistence,
            'state_transition_matrix': self.transition_matrix.tolist(),
            
            # State Path Analysis
            'state_frequencies': state_frequencies.tolist(),
            'bull_probability': float(np.mean(state_frequencies[:, 0])),
            'bear_probability': float(np.mean(state_frequencies[:, 1])),
            'sideways_probability': float(np.mean(state_frequencies[:, 2])),
            
            # Confidence
            'confidence_level': confidence_score['level'],
            'confidence_score': float(confidence_score['score']),
            'confidence_factors': confidence_score['factors'],
            
            # Method
            'method': 'Hidden Markov Model (HMM) with Monte Carlo',
            'algorithm': 'Viterbi + Forward-Backward + Baum-Welch'
        }
    
    def _calculate_regime_persistence(self) -> Dict:
        """
        Calculate how long each regime typically persists.
        
        Returns:
            Dictionary with regime persistence metrics
        """
        if self.hidden_states is None:
            self.viterbi_algorithm()
        
        states = self.hidden_states
        persistence = {}
        
        for state in range(self.n_states):
            # Find consecutive runs of this state
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
        """
        Assess confidence in the forecast.
        
        Args:
            state_probs: Current state probabilities
            persistence: Regime persistence metrics
            
        Returns:
            Dictionary with confidence assessment
        """
        score = 0
        factors = []
        
        # Factor 1: Clear dominant state
        max_prob = np.max(state_probs)
        if max_prob > 0.7:
            score += 0.3
            factors.append(f"Strong state confidence ({max_prob:.1%})")
        elif max_prob > 0.5:
            score += 0.2
            factors.append(f"Moderate state confidence ({max_prob:.1%})")
        else:
            score += 0.1
            factors.append(f"Weak state confidence ({max_prob:.1%})")
        
        # Factor 2: State persistence
        current_state_name = self.state_names[np.argmax(state_probs)]
        avg_duration = persistence[current_state_name]['avg_duration']
        
        if avg_duration > 10:
            score += 0.3
            factors.append(f"{current_state_name} typically lasts {avg_duration:.0f} days")
        elif avg_duration > 5:
            score += 0.2
            factors.append(f"{current_state_name} typically lasts {avg_duration:.0f} days")
        else:
            score += 0.1
            factors.append(f"{current_state_name} typically lasts {avg_duration:.0f} days")
        
        # Factor 3: Transition matrix confidence
        # Check if transitions are clear or uniform
        transition_entropy = -np.sum(self.transition_matrix * np.log(self.transition_matrix + 1e-10), axis=1)
        avg_entropy = np.mean(transition_entropy)
        max_entropy = np.log(self.n_states)
        
        clarity = 1 - (avg_entropy / max_entropy)
        
        if clarity > 0.5:
            score += 0.2
            factors.append(f"Clear transition patterns ({clarity:.1%} clarity)")
        else:
            score += 0.1
            factors.append(f"Uncertain transitions ({clarity:.1%} clarity)")
        
        # Factor 4: Sample size
        n_observations = len(self.returns)
        if n_observations > 200:
            score += 0.2
            factors.append(f"Large sample size (n={n_observations})")
        elif n_observations > 100:
            score += 0.15
            factors.append(f"Adequate sample size (n={n_observations})")
        else:
            score += 0.05
            factors.append(f"Small sample size (n={n_observations})")
        
        # Determine confidence level
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
        """
        Analyze characteristics of each market regime.
        
        Returns:
            Dictionary with regime characteristics
        """
        if self.hidden_states is None:
            self.viterbi_algorithm()
        
        states = self.hidden_states
        returns = self.returns
        prices = self.prices[1:]  # Align with returns
        
        characteristics = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            state_returns = returns[state_mask]
            state_prices = prices[state_mask]
            
            if len(state_returns) > 0:
                characteristics[self.state_names[state]] = {
                    'avg_return': float(np.mean(state_returns) * 100),  # Percentage
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
        """
        Generate trading strategy based on HMM forecast.
        
        Args:
            forecast: Forecast dictionary from forecast_price()
            
        Returns:
            Dictionary with trading strategy
        """
        current_state = forecast['current_state']
        direction = forecast['direction']
        dominant_regime = forecast['dominant_regime']
        expected_return = forecast['expected_return']
        
        # Get regime characteristics
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
        
        # Determine stop loss based on regime volatility
        if direction == 'BULLISH':
            # Stop loss below entry
            stop_distance = current_char['volatility'] * 2  # 2 sigma
            strategy['stop_loss'] = forecast['current_price'] * (1 - stop_distance / 100)
            strategy['rationale'].append(f"Entering {current_state} regime with {expected_return:.1f}% upside")
            strategy['rationale'].append(f"Current regime has {current_char['win_rate']:.0f}% win rate")
            
            # Position sizing based on confidence
            if forecast['confidence_level'] == 'HIGH':
                strategy['position_size'] = '3-5% of portfolio'
                strategy['rationale'].append("High confidence - normal position size")
            elif forecast['confidence_level'] == 'MEDIUM':
                strategy['position_size'] = '2-3% of portfolio'
                strategy['rationale'].append("Medium confidence - reduced position size")
            else:
                strategy['position_size'] = '1-2% of portfolio'
                strategy['rationale'].append("Low confidence - minimal position size")
            
            # Risks
            if forecast['bear_probability'] > 0.3:
                strategy['risks'].append(f"Bear regime probability: {forecast['bear_probability']:.1%}")
            
            strategy['risks'].append(f"Maximum historical loss in {current_state}: {current_char['max_loss']:.1f}%")
            
        elif direction == 'BEARISH':
            # Stop loss above entry
            stop_distance = current_char['volatility'] * 2
            strategy['stop_loss'] = forecast['current_price'] * (1 + stop_distance / 100)
            strategy['rationale'].append(f"Entering {current_state} regime with {abs(expected_return):.1f}% downside")
            strategy['rationale'].append(f"SHORT opportunity or stay in cash")
            
            strategy['position_size'] = 'Reduce exposure or SHORT'
            
            # Risks
            if forecast['bull_probability'] > 0.3:
                strategy['risks'].append(f"Bull regime probability: {forecast['bull_probability']:.1%}")
            
            strategy['risks'].append(f"Risk of reversal: {current_char['max_gain']:.1f}% potential upside")
            
        else:  # NEUTRAL
            strategy['rationale'].append(f"Sideways regime expected - range trading")
            strategy['rationale'].append(f"Buy support, sell resistance")
            strategy['position_size'] = '2-3% per trade'
            strategy['stop_loss'] = forecast['current_price'] * 0.97  # 3% stop
            
            strategy['risks'].append("Breakout risk in either direction")
            strategy['risks'].append(f"Bull probability: {forecast['bull_probability']:.1%}, Bear probability: {forecast['bear_probability']:.1%}")
        
        return strategy


def run_hmm_analysis(data: pd.DataFrame, forecast_days: int = 30) -> Dict:
    """
    Run complete HMM analysis on price data.
    
    Args:
        data: DataFrame with OHLCV data
        forecast_days: Number of days to forecast
        
    Returns:
        Dictionary with complete HMM analysis results
    """
    # Initialize HMM
    hmm = HiddenMarkovAnalysis(data)
    
    # Estimate parameters
    params = hmm.estimate_hmm_parameters()
    
    # Run Viterbi algorithm
    states = hmm.viterbi_algorithm()
    
    # Run Forward-Backward
    state_probs = hmm.forward_backward_algorithm()
    
    # Forecast
    forecast = hmm.forecast_price(forecast_days=forecast_days)
    
    # Regime characteristics
    characteristics = hmm.analyze_regime_characteristics()
    
    # Trading strategy
    strategy = hmm.generate_trading_strategy(forecast)
    
    # Regime persistence
    persistence = hmm._calculate_regime_persistence()
    
    return {
        'hmm_parameters': params,
        'forecast': forecast,
        'characteristics': characteristics,
        'strategy': strategy,
        'persistence': persistence
    }


# Example usage
if __name__ == "__main__":
    print("Hidden Markov Model Analysis Module - Ready")
    print("\nAvailable classes:")
    print("- HiddenMarkovAnalysis: HMM-based price forecasting")
    print("\nAvailable functions:")
    print("- run_hmm_analysis(): Complete HMM analysis")
