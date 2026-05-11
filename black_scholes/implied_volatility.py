# Init
from math import erf, exp, log, pi, sqrt

# Normal cumulative distribution function
def norm_cdf(x):

    return 0.5 * (1 + erf(x / sqrt(2)))

# Normal probability density function
def norm_pdf(x):

    return exp(-0.5 * (x ** 2)) / sqrt(2 * pi)

# The first term in the Black-Scholes formula, d1, is used in both the call price and vega calculations.
def d1(S, K, T, r, sigma):

    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

# The second term, d2, is used in the call price calculation and is derived from d1.
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)

# Black-Scholes call price formula.
def black_scholes_call(S, K, T, r, sigma):

    if sigma == 0:
        return max(S - K * exp(-r * T), 0)

    return S * norm_cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm_cdf(d2(S, K, T, r, sigma))

# Formula for vega, the sensitivity of the call price to changes in volatility.
def black_scholes_vega(S, K, T, r, sigma):
    
    if sigma == 0:
        return 0

    return S * sqrt(T) * norm_pdf(d1(S, K, T, r, sigma))

# --------------------------------
# Implied volatility by bisection
# --------------------------------

def implied_vol_bisection(target_price, S, K, T, r, sigma_low, sigma_high, price_tol=1e-10, vol_tol=1e-10, max_iter=500):

    price_low = black_scholes_call(S, K, T, r, sigma_low)
    price_high = black_scholes_call(S, K, T, r, sigma_high)

    while price_high < target_price and sigma_high < 20.0:
        sigma_high *= 2.0
        price_high = black_scholes_call(S, K, T, r, sigma_high)

    if price_low > target_price or price_high < target_price:
        raise RuntimeError("Could not bracket implied volatility. Try different sigma_low and sigma_high.")

    low = sigma_low
    high = sigma_high
    mid = None
    price_mid = None

    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (low + high)
        price_mid = black_scholes_call(S, K, T, r, mid)

        price_error = price_mid - target_price

        if abs(price_error) < price_tol or (high - low) < vol_tol:
            
            return mid

        if price_mid < target_price:
            
            low = mid
        
        else:
            
            high = mid

    return mid

# --------------------------------------
# Implied volatility by Newton's method
# --------------------------------------

def implied_vol_newton(target_price, S, K, T, r, initial_sigma=0.2, price_tol=1e-10, vol_tol=1e-10, max_iter=100):

    sigma = initial_sigma

    for iteration in range(1, max_iter + 1):
        price = black_scholes_call(S, K, T, r, sigma)
        price_error = price - target_price

        if abs(price_error) < price_tol:
            return sigma

        vega = black_scholes_vega(S, K, T, r, sigma)
        if vega > 0:
            sigma_next = sigma - price_error / vega
        if vega == 0:
            sigma_next = sigma + 0.001

        if abs(sigma_next - sigma) < vol_tol:
            sigma = sigma_next
            return sigma

        sigma = sigma_next

    return sigma


# ------------
# Test case
# ------------

S=100
K=140
T=1
r=0.04

sigma_true = 0.25
computed_price = black_scholes_call(S, K, T, r, sigma_true) # Compute the price from true sigma

sigma_bisect = implied_vol_bisection(computed_price, S, K, T, r, sigma_low=0.01, sigma_high=1.0) # Try to recover the sigma from the price using bisection
sigma_newton = implied_vol_newton(computed_price, S, K, T, r, initial_sigma=0.2) # Try to recover the sigma from the price using Newton's method

print(f"S={S}")
print(f"K={K}")
print(f"T={T}")
print(f"r={r}")
print(f"True sigma={sigma_true}")
print(f"Black-Scholes price={computed_price}")
print(f"Bisection IV={sigma_bisect}")
print(f"Newton IV={sigma_newton}")

# Test different initial guesses for Newton's method
for initial_sigma in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1]:
    sigma_newton = implied_vol_newton(computed_price, S, K, T, r, initial_sigma=initial_sigma)
    
    print(f"Initial sigma={initial_sigma}, True sigma={sigma_true}, Newton IV={sigma_newton:.4f}")

# Test different values of K
for K in [100, 120, 150, 200, 250, 300, 325, 350, 375, 400]:
    computed_price = black_scholes_call(S, K, T, r, sigma_true)
    sigma_bisect = implied_vol_bisection(computed_price, S, K, T, r, sigma_low=0.01, sigma_high=1.0)
    sigma_newton = implied_vol_newton(computed_price, S, K, T, r, initial_sigma=0.2)

    print(f"K={K}, Price={computed_price:.4f}, Bisection IV={sigma_bisect:.4f}, Newton IV={sigma_newton:.4f}")
