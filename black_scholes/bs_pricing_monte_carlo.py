# Init
import numpy as np
from math import exp, sqrt, erf

# Seed
np.random.seed(4321)

# CDF of standard normal distribution using the error function.
def normal_cdf(x):

    return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))

# Closed-form Black-Scholes formula for a European call option.
def european_formula(S, K, T, r, sigma):

    S = np.asarray(S, dtype=float)
    d1 = np.zeros_like(S)
    d2 = np.zeros_like(S)
    price = np.zeros_like(S)

    # Use formula (4) in pdf
    d1[S>0] = (np.log(S[S>0] / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2[S>0] = d1[S>0] - sigma * np.sqrt(T)
    price[S>0] = S[S>0] * normal_cdf(d1[S>0])- K * np.exp(-r * T) * normal_cdf(d2[S>0])

    return price

# Closed-form cash-or-nothing digital call price
def black_scholes_digital_call(S, K, T, r, sigma):

    S = np.asarray(S, dtype=float)
    d2 = np.zeros_like(S)
    price = np.zeros_like(S)

    d2[S>0] = (np.log(S[S>0] / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    price[S>0] = np.exp(-r * T) * normal_cdf(d2[S>0])

    return price

# MC pricing using antithetic variates for both European and digital call options.
def monte_carlo_black_scholes(S0, K, T, r, sigma, n_paths=1000000):

    # Simulate under the risk-neutral measure Q: dS_t = r S_t dt + sigma S_t dW_t^Q
    Z = np.random.standard_normal(n_paths)

    ST_plus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * Z)
    ST_minus = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * sqrt(T) * Z)

    # Payoffs
    call_payoff_plus = np.maximum(ST_plus - K, 0.0)
    call_payoff_minus = np.maximum(ST_minus - K, 0.0)
    call_payoff = 0.5 * (call_payoff_plus + call_payoff_minus)

    digital_payoff_plus = (ST_plus > K).astype(float)
    digital_payoff_minus = (ST_minus > K).astype(float)
    digital_payoff = 0.5 * (digital_payoff_plus + digital_payoff_minus)

    discount = exp(-r * T)
    call_price = discount * np.mean(call_payoff)
    digital_price = discount * np.mean(digital_payoff)

    # Standard errors
    european_se = discount * np.std(call_payoff) / sqrt(n_paths)
    digital_se = discount * np.std(digital_payoff) / sqrt(n_paths)

    return call_price, european_se, digital_price, digital_se

#############################################################

# Test case

if __name__ == "__main__":

    S0 = 100      
    K = 130       
    T = 1  
    r = 0.05
    sigma = 0.20

    european_exact = european_formula(S0, K, T, r, sigma)
    digital_exact = black_scholes_digital_call(S0, K, T, r, sigma)

    european_mc, european_se, digital_mc, digital_se = monte_carlo_black_scholes(S0, K, T, r, sigma)

    print("European call option")
    print(f"Monte Carlo price: {european_mc:.6f}")
    print(f"Standard error:    {european_se:.6f}")
    print(f"Closed-form price: {european_exact:.6f}")

    print("Digital call option")
    print(f"Monte Carlo price: {digital_mc:.6f}")
    print(f"Standard error:    {digital_se:.6f}")
    print(f"Closed-form price: {digital_exact:.6f}")
