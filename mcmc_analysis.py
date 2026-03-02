"""
Markov Chain Monte Carlo (MCMC) Analysis for Stock Price Forecasting
=====================================================================

This module implements a full Bayesian MCMC pipeline for equity price forecasting:

  1. LIKELIHOOD  — Log-normal return model (Geometric Brownian Motion)
  2. PRIORS      — Weakly informative Normal / Half-Normal priors on μ, σ
  3. SAMPLER     — Metropolis-Hastings with adaptive step-size tuning
  4. DIAGNOSTICS — R-hat (Gelman-Rubin), effective sample size, trace analysis
  5. SIMULATION  — Monte Carlo price fans using posterior predictive draws
  6. RISK        — VaR, CVaR, Probability of Profit, Sharpe (posterior)

Key difference from plain Monte Carlo:
  • Plain MC fixes μ and σ as point estimates → single "best guess" forecast
  • MCMC samples the FULL posterior P(μ, σ | data) → honest uncertainty
    that widens/narrows with data quality, capturing parameter uncertainty
    (epistemic) on top of market randomness (aleatoric).

Author: Market Analyzer Pro — MCMC Bayesian Forecasting Module
Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1 — BAYESIAN MODEL (likelihood + priors)
# ============================================================================

def log_likelihood(log_returns: np.ndarray, mu: float, sigma: float) -> float:
    """
    Log-likelihood under Geometric Brownian Motion (GBM).

    Log-returns are assumed i.i.d. Normal(μ - σ²/2, σ²).
    We work with the 'drift-adjusted' parameterisation so μ is the
    instantaneous expected return (annualised-compatible).

    Args:
        log_returns : array of daily log returns
        mu          : daily drift  (parameter to infer)
        sigma       : daily volatility (parameter to infer)

    Returns:
        Scalar log-likelihood value (−∞ if σ ≤ 0)
    """
    if sigma <= 0:
        return -np.inf

    # GBM: r_t ~ N(μ - 0.5 σ², σ²)
    drift = mu - 0.5 * sigma ** 2
    ll = stats.norm.logpdf(log_returns, loc=drift, scale=sigma).sum()
    return float(ll)


def log_prior(mu: float, sigma: float) -> float:
    """
    Log-prior: weakly informative conjugate-style priors.

    μ ~ Normal(0, 0.1)    — centred on zero daily drift, ±10 % daily is wide
    σ ~ Half-Normal(0.03) — most daily vol lives in [0.005, 0.08]

    Returns −∞ for impossible parameter values (σ ≤ 0).
    """
    if sigma <= 0:
        return -np.inf

    lp_mu    = stats.norm.logpdf(mu,    loc=0.0,  scale=0.10)
    lp_sigma = stats.halfnorm.logpdf(sigma, loc=0.0, scale=0.03)
    return float(lp_mu + lp_sigma)


def log_posterior(log_returns: np.ndarray, mu: float, sigma: float) -> float:
    """Unnormalised log-posterior = log-likelihood + log-prior."""
    return log_likelihood(log_returns, mu, sigma) + log_prior(mu, sigma)


# ============================================================================
# SECTION 2 — METROPOLIS-HASTINGS SAMPLER
# ============================================================================

class MetropolisHastingsSampler:
    """
    Random-walk Metropolis-Hastings with per-parameter adaptive step sizes.

    Tuning:
        • Target acceptance rate ≈ 0.234 (optimal for 2-D Gaussian targets)
        • Step size adapted every `tune_interval` steps during warm-up
        • After warm-up the step sizes are frozen (ensures Markov property)
    """

    def __init__(
        self,
        log_returns: np.ndarray,
        n_samples:   int   = 5_000,
        n_warmup:    int   = 2_000,
        n_chains:    int   = 4,
        tune_interval: int = 100,
        target_accept: float = 0.234,
        seed: int = 42,
    ):
        self.log_returns   = log_returns
        self.n_samples     = n_samples
        self.n_warmup      = n_warmup
        self.n_chains      = n_chains
        self.tune_interval = tune_interval
        self.target_accept = target_accept
        self.rng           = np.random.default_rng(seed)

        # Initialise step sizes
        self.step_mu    = 0.005
        self.step_sigma = 0.002

    # ── Single-chain sampler ────────────────────────────────────────────────

    def _run_single_chain(
        self,
        init_mu:    float,
        init_sigma: float,
        chain_id:   int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run one MCMC chain.

        Returns
        -------
        mu_samples    : (n_samples,) array
        sigma_samples : (n_samples,) array
        accept_rate   : overall acceptance rate
        """
        rng = np.random.default_rng(self.rng.integers(0, 2**31) + chain_id)

        # Storage (warmup + samples)
        total = self.n_warmup + self.n_samples
        mu_arr    = np.empty(total)
        sigma_arr = np.empty(total)

        mu_cur    = init_mu
        sigma_cur = init_sigma
        lp_cur    = log_posterior(self.log_returns, mu_cur, sigma_cur)

        step_mu    = self.step_mu
        step_sigma = self.step_sigma
        accept_count = 0
        tune_accepts = 0

        for i in range(total):
            # Propose new parameters (independent RW for each)
            mu_prop    = mu_cur    + rng.normal(0, step_mu)
            sigma_prop = sigma_cur + rng.normal(0, step_sigma)

            lp_prop = log_posterior(self.log_returns, mu_prop, sigma_prop)

            # Acceptance ratio (log-space for numerical stability)
            log_alpha = lp_prop - lp_cur
            if np.log(rng.uniform()) < log_alpha:
                mu_cur    = mu_prop
                sigma_cur = sigma_prop
                lp_cur    = lp_prop
                accept_count += 1
                if i < self.n_warmup:
                    tune_accepts += 1

            mu_arr[i]    = mu_cur
            sigma_arr[i] = sigma_cur

            # Adaptive tuning during warm-up
            if i < self.n_warmup and (i + 1) % self.tune_interval == 0:
                rate = tune_accepts / self.tune_interval
                # Scale step toward target acceptance rate
                factor = np.exp(rate - self.target_accept)
                factor = np.clip(factor, 0.5, 2.0)
                step_mu    *= factor
                step_sigma *= factor
                tune_accepts = 0

        # Discard warm-up
        mu_post    = mu_arr[self.n_warmup:]
        sigma_post = sigma_arr[self.n_warmup:]
        final_accept = accept_count / total

        return mu_post, sigma_post, final_accept

    # ── Multi-chain runner ──────────────────────────────────────────────────

    def sample(self) -> Dict:
        """
        Run all chains in sequence (no threading for portability).

        Returns a dict with:
          mu_samples, sigma_samples : (n_chains × n_samples) arrays
          acceptance_rates          : per-chain accept rates
          r_hat_mu, r_hat_sigma     : Gelman-Rubin convergence diagnostics
          ess_mu, ess_sigma         : effective sample sizes
        """
        # Dispersed starting points drawn from prior
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        lr  = self.log_returns
        em  = lr.mean()
        es  = lr.std()

        init_mus    = rng.normal(em, 0.5 * abs(em) + 1e-4, self.n_chains)
        init_sigmas = np.abs(rng.normal(es, 0.3 * es + 1e-4, self.n_chains)) + 1e-5

        mu_chains    = np.empty((self.n_chains, self.n_samples))
        sigma_chains = np.empty((self.n_chains, self.n_samples))
        accept_rates = []

        for c in range(self.n_chains):
            mu_c, sigma_c, ar = self._run_single_chain(
                init_mus[c], init_sigmas[c], chain_id=c)
            mu_chains[c]    = mu_c
            sigma_chains[c] = sigma_c
            accept_rates.append(ar)

        # Pool samples for downstream use
        mu_flat    = mu_chains.flatten()
        sigma_flat = sigma_chains.flatten()

        return {
            'mu_samples':      mu_flat,
            'sigma_samples':   sigma_flat,
            'mu_chains':       mu_chains,
            'sigma_chains':    sigma_chains,
            'acceptance_rates': accept_rates,
            'r_hat_mu':        self._gelman_rubin(mu_chains),
            'r_hat_sigma':     self._gelman_rubin(sigma_chains),
            'ess_mu':          self._effective_sample_size(mu_chains),
            'ess_sigma':       self._effective_sample_size(sigma_chains),
            'n_samples':       self.n_samples,
            'n_chains':        self.n_chains,
            'n_warmup':        self.n_warmup,
        }

    # ── Convergence diagnostics ─────────────────────────────────────────────

    @staticmethod
    def _gelman_rubin(chains: np.ndarray) -> float:
        """
        Gelman-Rubin R-hat statistic (should be < 1.01 for convergence).

        chains : (n_chains, n_samples)
        """
        n_chains, n = chains.shape
        if n_chains < 2:
            return 1.0

        chain_means = chains.mean(axis=1)          # (n_chains,)
        chain_vars  = chains.var(axis=1, ddof=1)   # (n_chains,)

        overall_mean = chain_means.mean()

        # Between-chain variance B
        B = n * chain_means.var(ddof=1)
        # Within-chain variance W
        W = chain_vars.mean()

        # Pooled posterior variance estimate
        var_hat = (1 - 1/n) * W + (1/n) * B
        r_hat   = np.sqrt(var_hat / W) if W > 0 else np.inf
        return float(r_hat)

    @staticmethod
    def _effective_sample_size(chains: np.ndarray) -> float:
        """
        Bulk effective sample size via autocorrelation.

        chains : (n_chains, n_samples)
        """
        n_chains, n = chains.shape
        # Pool and estimate lag-k autocorrelations
        samples = chains.flatten()
        if samples.std() < 1e-10:
            return float(len(samples))

        # Autocorrelation up to lag 100
        max_lag = min(100, n // 4)
        acf = []
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
            if np.isnan(c):
                break
            acf.append(c)
            if c < 0:  # Geyer's initial monotone sequence
                break

        rho_sum = 1 + 2 * sum(acf)
        ess = len(samples) / max(rho_sum, 1.0)
        return float(ess)


# ============================================================================
# SECTION 3 — POSTERIOR PREDICTIVE PRICE SIMULATION
# ============================================================================

def simulate_price_paths(
    current_price: float,
    mu_samples:    np.ndarray,
    sigma_samples: np.ndarray,
    forecast_days: int = 30,
    n_paths:       int = 2_000,
    seed:          int = 99,
) -> np.ndarray:
    """
    Generate Monte Carlo price paths by drawing (μ, σ) from the MCMC posterior.

    Each path uses a DIFFERENT (μ, σ) pair → captures both parameter
    uncertainty (epistemic) and daily noise (aleatoric), giving wider,
    more honest forecast intervals than fixed-parameter simulation.

    Returns
    -------
    paths : (n_paths, forecast_days) array of simulated prices
    """
    rng  = np.random.default_rng(seed)
    n    = len(mu_samples)
    idx  = rng.integers(0, n, size=n_paths)        # sample with replacement

    mu_draw    = mu_samples[idx]                    # (n_paths,)
    sigma_draw = sigma_samples[idx]                 # (n_paths,)

    # Daily log-returns: r_t ~ N(μ - 0.5σ², σ²)
    drift     = mu_draw - 0.5 * sigma_draw ** 2    # (n_paths,)
    noise     = rng.normal(0, 1, size=(n_paths, forecast_days))
    log_ret   = drift[:, None] + sigma_draw[:, None] * noise

    # Cumulative product of exponentiated returns
    paths = current_price * np.exp(np.cumsum(log_ret, axis=1))
    return paths                                    # (n_paths, forecast_days)


# ============================================================================
# SECTION 4 — RISK METRICS FROM POSTERIOR
# ============================================================================

def compute_risk_metrics(
    paths:         np.ndarray,
    current_price: float,
    horizon:       int = 30,
    confidence:    float = 0.95,
) -> Dict:
    """
    Compute standard and Bayesian risk metrics from simulated paths.

    Parameters
    ----------
    paths         : (n_paths, forecast_days) price array
    current_price : today's price
    horizon       : which day to evaluate (default = last day)
    confidence    : VaR/CVaR confidence level (default 0.95)
    """
    terminal_prices  = paths[:, horizon - 1]
    terminal_returns = (terminal_prices - current_price) / current_price

    # ── Basic statistics ──────────────────────────────────────────────────
    mean_ret   = float(np.mean(terminal_returns))
    median_ret = float(np.median(terminal_returns))
    std_ret    = float(np.std(terminal_returns))

    # ── VaR & CVaR ───────────────────────────────────────────────────────
    alpha     = 1 - confidence
    var_level = float(np.percentile(terminal_returns, alpha * 100))
    cvar_lvl  = float(terminal_returns[terminal_returns <= var_level].mean()) \
                if (terminal_returns <= var_level).any() else var_level

    # ── Probability of profit ─────────────────────────────────────────────
    prob_profit = float(np.mean(terminal_returns > 0))
    prob_5pct   = float(np.mean(terminal_returns > 0.05))
    prob_10pct  = float(np.mean(terminal_returns > 0.10))
    prob_loss5  = float(np.mean(terminal_returns < -0.05))

    # ── Credible intervals ────────────────────────────────────────────────
    ci_50  = (float(np.percentile(terminal_prices, 25)),
               float(np.percentile(terminal_prices, 75)))
    ci_80  = (float(np.percentile(terminal_prices, 10)),
               float(np.percentile(terminal_prices, 90)))
    ci_95  = (float(np.percentile(terminal_prices, 2.5)),
               float(np.percentile(terminal_prices, 97.5)))

    # ── Fan chart percentile bands (time series) ──────────────────────────
    pcts = [2.5, 10, 25, 50, 75, 90, 97.5]
    fan  = {str(p): paths[:, :horizon].T.tolist() for p in pcts}  # placeholder
    bands = {}
    for p in pcts:
        bands[str(p)] = np.percentile(paths[:, :horizon], p, axis=0).tolist()

    return {
        # Central tendency
        'mean_return':   mean_ret,
        'median_return': median_ret,
        'std_return':    std_ret,
        'mean_price':    float(np.mean(terminal_prices)),
        'median_price':  float(np.median(terminal_prices)),

        # Downside risk
        f'var_{int(confidence*100)}':  var_level,
        f'cvar_{int(confidence*100)}': cvar_lvl,

        # Probabilities
        'prob_profit':    prob_profit,
        'prob_gain_5pct': prob_5pct,
        'prob_gain_10pct':prob_10pct,
        'prob_loss_5pct': prob_loss5,

        # Credible intervals (prices)
        'ci_50':  ci_50,
        'ci_80':  ci_80,
        'ci_95':  ci_95,

        # Fan bands for chart (day-by-day percentiles)
        'fan_bands': bands,

        # Raw paths (subset for visualisation)
        'sample_paths': paths[:200, :horizon],
    }


# ============================================================================
# SECTION 5 — POSTERIOR SUMMARY
# ============================================================================

def summarise_posterior(
    mcmc_result:  Dict,
    log_returns:  np.ndarray,
) -> Dict:
    """
    Build a human-readable summary of the MCMC posterior.

    Includes:
      • Point estimates (mean, median, MAP approximation)
      • Credible intervals for μ and σ
      • Convergence diagnostics
      • Comparison with MLE (frequentist) estimates
    """
    mu    = mcmc_result['mu_samples']
    sigma = mcmc_result['sigma_samples']

    # ── Posterior statistics ──────────────────────────────────────────────
    def summarise(arr, name):
        return {
            f'{name}_mean':        float(arr.mean()),
            f'{name}_median':      float(np.median(arr)),
            f'{name}_std':         float(arr.std()),
            f'{name}_ci_90_lo':    float(np.percentile(arr, 5)),
            f'{name}_ci_90_hi':    float(np.percentile(arr, 95)),
            f'{name}_ci_95_lo':    float(np.percentile(arr, 2.5)),
            f'{name}_ci_95_hi':    float(np.percentile(arr, 97.5)),
        }

    post_mu    = summarise(mu,    'mu')
    post_sigma = summarise(sigma, 'sigma')

    # ── MLE (frequentist baseline) ────────────────────────────────────────
    mle_mu_adj   = float(log_returns.mean())          # adjusted drift
    mle_mu_true  = mle_mu_adj + 0.5 * log_returns.var()
    mle_sigma    = float(log_returns.std())

    # ── Convergence ───────────────────────────────────────────────────────
    r_hat_mu    = mcmc_result['r_hat_mu']
    r_hat_sigma = mcmc_result['r_hat_sigma']
    ess_mu      = mcmc_result['ess_mu']
    ess_sigma   = mcmc_result['ess_sigma']

    converged = (r_hat_mu < 1.05) and (r_hat_sigma < 1.05) \
                and (ess_mu > 400) and (ess_sigma > 400)

    avg_accept = float(np.mean(mcmc_result['acceptance_rates']))

    # ── Annualised versions (×252 trading days for μ, ×√252 for σ) ────────
    ann_mu_mean  = post_mu['mu_mean']    * 252
    ann_mu_lo95  = post_mu['mu_ci_95_lo'] * 252
    ann_mu_hi95  = post_mu['mu_ci_95_hi'] * 252
    ann_sig_mean = post_sigma['sigma_mean']  * np.sqrt(252)

    return {
        **post_mu,
        **post_sigma,

        # MLE comparison
        'mle_mu_daily':    mle_mu_adj,
        'mle_mu_true':     mle_mu_true,
        'mle_sigma_daily': mle_sigma,

        # Annualised
        'ann_mu_mean':  ann_mu_mean,
        'ann_mu_lo95':  ann_mu_lo95,
        'ann_mu_hi95':  ann_mu_hi95,
        'ann_sigma':    ann_sig_mean,

        # Convergence
        'r_hat_mu':        r_hat_mu,
        'r_hat_sigma':     r_hat_sigma,
        'ess_mu':          ess_mu,
        'ess_sigma':       ess_sigma,
        'avg_accept_rate': avg_accept,
        'converged':       converged,

        # Meta
        'n_obs':           len(log_returns),
        'n_total_samples': mcmc_result['n_samples'] * mcmc_result['n_chains'],
    }


# ============================================================================
# SECTION 6 — MAIN ENTRY POINT
# ============================================================================

def run_mcmc_analysis(
    data:          pd.DataFrame,
    forecast_days: int  = 30,
    n_samples:     int  = 4_000,
    n_warmup:      int  = 1_500,
    n_chains:      int  = 4,
    n_paths:       int  = 2_000,
    seed:          int  = 42,
) -> Dict:
    """
    Full MCMC Bayesian analysis pipeline.

    Parameters
    ----------
    data          : DataFrame with at least a 'Close' column
    forecast_days : how many trading days to forecast
    n_samples     : posterior draws per chain (after warmup)
    n_warmup      : discarded warmup steps per chain
    n_chains      : parallel chains (run sequentially here)
    n_paths       : Monte Carlo paths for price simulation
    seed          : reproducibility seed

    Returns
    -------
    Dictionary with keys:
      posterior, risk_metrics, price_paths, forecast_summary, diagnostics
    """
    # ── Prepare returns ───────────────────────────────────────────────────
    close       = data['Close'].dropna().values
    log_returns = np.diff(np.log(close))

    current_price = float(close[-1])

    # ── Run MCMC ──────────────────────────────────────────────────────────
    sampler = MetropolisHastingsSampler(
        log_returns  = log_returns,
        n_samples    = n_samples,
        n_warmup     = n_warmup,
        n_chains     = n_chains,
        seed         = seed,
    )
    mcmc_result = sampler.sample()

    # ── Posterior summary ─────────────────────────────────────────────────
    posterior = summarise_posterior(mcmc_result, log_returns)

    # ── Simulate paths using posterior draws ──────────────────────────────
    paths = simulate_price_paths(
        current_price = current_price,
        mu_samples    = mcmc_result['mu_samples'],
        sigma_samples = mcmc_result['sigma_samples'],
        forecast_days = forecast_days,
        n_paths       = n_paths,
        seed          = seed,
    )

    # ── Risk metrics ──────────────────────────────────────────────────────
    risk = compute_risk_metrics(paths, current_price, horizon=forecast_days)

    # ── High-level forecast summary ───────────────────────────────────────
    target_price = risk['median_price']
    exp_return   = risk['median_return']

    if exp_return > 0.05:
        direction, signal = 'BULLISH', 'BUY'
    elif exp_return < -0.05:
        direction, signal = 'BEARISH', 'SELL'
    else:
        direction, signal = 'NEUTRAL', 'HOLD'

    # ── Forecast dates ────────────────────────────────────────────────────
    try:
        last_date    = data.index[-1]
        fdates       = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                       periods=forecast_days)
        forecast_dates = [d.to_pydatetime() for d in fdates]
    except Exception:
        last_date = pd.Timestamp.now()
        forecast_dates = [last_date + pd.Timedelta(days=i+1)
                          for i in range(forecast_days)]

    # ── Regime characterisation from posterior ────────────────────────────
    # Use posterior μ sign to gauge "likely regime"
    mu_pos_frac = float(np.mean(mcmc_result['mu_samples'] > 0))
    if mu_pos_frac > 0.70:
        regime = 'Bullish Drift Regime'
    elif mu_pos_frac < 0.30:
        regime = 'Bearish Drift Regime'
    else:
        regime = 'Uncertain / Mean-Reverting Regime'

    forecast_summary = {
        'current_price':    current_price,
        'target_price':     target_price,
        'mean_price':       risk['mean_price'],
        'ci_95_low':        risk['ci_95'][0],
        'ci_95_high':       risk['ci_95'][1],
        'ci_80_low':        risk['ci_80'][0],
        'ci_80_high':       risk['ci_80'][1],
        'ci_50_low':        risk['ci_50'][0],
        'ci_50_high':       risk['ci_50'][1],
        'expected_return':  exp_return * 100,
        'direction':        direction,
        'signal':           signal,
        'regime':           regime,
        'mu_pos_fraction':  mu_pos_frac,
        'forecast_days':    forecast_days,
        'forecast_dates':   forecast_dates,
        'fan_bands':        risk['fan_bands'],
        'sample_paths':     risk['sample_paths'],
        # Annualised posterior params
        'ann_drift_mean':   posterior['ann_mu_mean']  * 100,
        'ann_drift_lo':     posterior['ann_mu_lo95']  * 100,
        'ann_drift_hi':     posterior['ann_mu_hi95']  * 100,
        'ann_volatility':   posterior['ann_sigma']    * 100,
    }

    diagnostics = {
        'converged':       posterior['converged'],
        'r_hat_mu':        posterior['r_hat_mu'],
        'r_hat_sigma':     posterior['r_hat_sigma'],
        'ess_mu':          posterior['ess_mu'],
        'ess_sigma':       posterior['ess_sigma'],
        'accept_rate':     posterior['avg_accept_rate'],
        'n_obs':           posterior['n_obs'],
        'n_total_samples': posterior['n_total_samples'],
    }

    return {
        'mcmc_result':      mcmc_result,
        'posterior':        posterior,
        'risk_metrics':     risk,
        'forecast_summary': forecast_summary,
        'diagnostics':      diagnostics,
        'log_returns':      log_returns,
    }


# ============================================================================
# SECTION 7 — STANDALONE TEST
# ============================================================================

if __name__ == '__main__':
    print("MCMC Analysis Module — self-test")

    # Synthetic data
    rng   = np.random.default_rng(0)
    true_mu, true_sigma = 0.0005, 0.015
    n     = 252
    rets  = rng.normal(true_mu - 0.5 * true_sigma**2, true_sigma, n)
    prices = 1000 * np.exp(np.cumsum(rets))
    df    = pd.DataFrame({'Close': prices})
    df.index = pd.bdate_range('2024-01-01', periods=n)

    result = run_mcmc_analysis(df, n_samples=1000, n_warmup=500,
                                n_chains=2, n_paths=500)

    post = result['posterior']
    diag = result['diagnostics']
    fs   = result['forecast_summary']

    print(f"\nPosterior μ (daily): {post['mu_mean']:.6f}  "
          f"[True: {true_mu:.6f}]")
    print(f"Posterior σ (daily): {post['sigma_mean']:.6f}  "
          f"[True: {true_sigma:.6f}]")
    print(f"R-hat μ: {diag['r_hat_mu']:.4f}  "
          f"R-hat σ: {diag['r_hat_sigma']:.4f}")
    print(f"ESS μ: {diag['ess_mu']:.0f}  ESS σ: {diag['ess_sigma']:.0f}")
    print(f"Accept rate: {diag['accept_rate']:.2%}")
    print(f"Converged: {diag['converged']}")
    print(f"\n30-day Forecast: {fs['direction']} | "
          f"Target ₹{fs['target_price']:.2f} | "
          f"Return {fs['expected_return']:.2f}%")
    print(f"95% CI: ₹{fs['ci_95_low']:.2f} – ₹{fs['ci_95_high']:.2f}")
