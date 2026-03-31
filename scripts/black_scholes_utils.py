import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes(S, K, T, r, sigma, option_type='CE'):
    """
    Standard Black-Scholes formula.
    S: Spot Price
    K: Strike Price
    T: Time to Expiry (in years)
    r: Risk-free rate (e.g. 0.07 for 7%)
    sigma: Volatility (e.g. 0.20 for 20%)
    """
    if T <= 0 or sigma <= 0:
        if option_type == 'CE':
            return max(0, S - K)
        else:
            return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'CE':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_iv(price, S, K, T, r, option_type='CE'):
    """
    Back-calculate Implied Volatility (IV) using Brent's method.
    """
    if T <= 0 or price <= 0:
        return 0.0
    
    # Objective function: price - BS(sigma) = 0
    def objective(sigma):
        return price - black_scholes(S, K, T, r, sigma, option_type)
    
    try:
        # Search for sigma between 0.0001% and 500%
        iv = brentq(objective, 1e-6, 5.0)
        return iv
    except (ValueError, RuntimeError):
        # If no solution found, try a simpler approach or return 0
        return 0.0

def calculate_greeks(S, K, T, r, sigma, option_type='CE'):
    """
    Calculate Greeks: Delta, Gamma, Theta, Vega.
    """
    if T <= 0 or sigma <= 1e-4:
        return {
            'delta': 1.0 if (option_type == 'CE' and S > K) else (-1.0 if (option_type == 'PE' and S < K) else 0.0),
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # N'(d1) is the probability density function (standard normal)
    n_prime_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)

    # 1. Delta
    if option_type == 'CE':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    
    # 2. Gamma (Same for Call and Put)
    gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
    
    # 3. Vega (Same for Call and Put)
    # Vega is often expressed as 'per 1% change', but we return raw BS Vega
    vega = S * n_prime_d1 * np.sqrt(T) / 100.0 # dividing by 100 to get value per 1 vol point
    
    # 4. Theta (Annualized, convert to per-day if needed)
    if option_type == 'CE':
        theta = (- (S * n_prime_d1 * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        theta = (- (S * n_prime_d1 * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    # Convert Theta to per-day (Assuming 365 days)
    theta_per_day = theta / 365.0
    
    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta_per_day),
        'vega': float(vega)
    }
