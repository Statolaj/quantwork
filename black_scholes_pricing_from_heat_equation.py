# Init
import numpy as np
import matplotlib.pyplot as plt
from math import erf

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

# Closed-form Black-Scholes formula for a digital call option.
def digital_formula(S, K, T, r, sigma):

    S = np.asarray(S, dtype=float)
    d2 = np.zeros_like(S)
    price = np.zeros_like(S)
    
    # Use formula (5) in pdf
    d2[S>0] = (np.log(S[S>0] / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    price[S>0] = np.exp(-r * T) * normal_cdf(d2[S>0])

    return price

# Function as input takes stock price, strike, time to maturity, risk-free rate, volatility, number of space grid points and if option type is "european" or "digital".
# Returns the price of the option computed from the heat equation method, the grid of log-prices, the final solution h at time T.
def solve_black_scholes_from_heat(S0, K, T, r, sigma, option_type, n_space):
    
    alpha = 0.5 * sigma**2
    b = r - 0.5 * sigma**2

    # Put log(K) exactly on the center grid point.
    x_K = np.log(K)
    x_target = np.log(S0) + b * T # From formula (3) in pdf.
    half_width = abs(x_target - x_K) + 5 * sigma
    x_min = x_K - half_width
    x_max = x_K + half_width
    
    if n_space % 2 == 0:
            n_space += 1

    x_grid = np.linspace(x_min, x_max, n_space)
    delta = x_grid[1] - x_grid[0]

    # European case
    if option_type == "european":
        if n_space % 2 == 0:
            n_space += 1
        middle_index = n_space // 2
        h = np.maximum(np.exp(x_grid) - K, 0.0)

    # Digital case
    if option_type == "digital":
        if n_space % 2 == 1:
            n_space += 1
        left_index = n_space // 2 - 1
        right_index = n_space // 2
        midpoint = 0.5 * (x_grid[left_index] + x_grid[right_index])
        h = (x_grid > x_K).astype(float)
        
    # Stability condition for the explicit finite difference method
    delta_t = 0.45 * delta**2 / (2 * alpha)
    n_steps = int(np.ceil(T / delta_t))
    delta_t = T / n_steps # So delta_t divides T exactly

    print(f"delta = {delta}"), print(f"delta_t = {delta_t}"), print(f"Number of time steps = {n_steps}")

   # Algorithm running the finite difference method for n_steps time steps
    for n in range(n_steps):
        tau_new = (n + 1) * delta_t

        h_new = h.copy()
        h_xx = (h[2:] - 2.0 * h[1:-1] + h[:-2]) / delta**2
        h_new[1:-1] = h[1:-1] + delta_t * alpha * h_xx

        # Lower boundary: far below strike
        h_new[0] = 0.0

        # Upper boundary depends on payoff
        if option_type == "european":
            h_new[-1] = np.exp(x_max + 0.5 * sigma**2 * tau_new) - K

        elif option_type == "digital":
            h_new[-1] = 1.0

        h = h_new

    h_target = np.interp(x_target, x_grid, h)
    price = np.exp(-r * T) * h_target

    return price, x_grid, h



####################################

########## Test example ############

####################################

S0, K, T, r, sigma, n_space = 200, 220, 1, 0.05, 0.5, 1000
b = r - 0.5 * sigma**2

# Solve with heat-equation method
price_eur, x_eur, h_eur = solve_black_scholes_from_heat(S0, K, T, r, sigma, "european", n_space)
price_dig, x_dig, h_dig = solve_black_scholes_from_heat(S0, K, T, r, sigma, "digital", n_space)

# Closed-form prices at S0
closed_eur = european_formula(np.array([S0], dtype=float), K, T, r, sigma)[0]
closed_dig = digital_formula(np.array([S0], dtype=float), K, T, r, sigma)[0]

# Transform heat solutions back to stock-price space
S_eur = np.exp(x_eur - b * T)
S_dig = np.exp(x_dig - b * T)

V_eur_heat = np.exp(-r * T) * h_eur
V_dig_heat = np.exp(-r * T) * h_dig

V_eur_closed = european_formula(S_eur, K, T, r, sigma)
V_dig_closed = digital_formula(S_dig, K, T, r, sigma)

# Print price comparison
print(f"{'Option':<15}{'Heat PDE':>12}{'Closed form':>15}{'Abs error':>15}")
print(f"{'European':<15}{price_eur:>12.6f}{closed_eur:>15.6f}{abs(price_eur - closed_eur):>15.6f}")
print(f"{'Digital':<15}{price_dig:>12.6f}{closed_dig:>15.6f}{abs(price_dig - closed_dig):>15.6f}")

# Plot range
S_min, S_max = 0, 2 * K
range_eur = (S_eur >= S_min) & (S_eur <= S_max)
range_dig = (S_dig >= S_min) & (S_dig <= S_max)

# Plot side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(S_eur[range_eur], V_eur_heat[range_eur], label="Heat PDE")
ax[0].plot(S_eur[range_eur], V_eur_closed[range_eur], "--", label="Closed form")
ax[0].axvline(K, color="black", linestyle=":", label="Strike K")
ax[0].set(xlim=(S_min, S_max), xlabel="Stock price S", ylabel="Call price", title="European call")
ax[0].legend()

ax[1].plot(S_dig[range_dig], V_dig_heat[range_dig], label="Heat PDE")
ax[1].plot(S_dig[range_dig], V_dig_closed[range_dig], "--", label="Closed form")
ax[1].axvline(K, color="black", linestyle=":", label="Strike K")
ax[1].set(xlim=(S_min, S_max), ylim=(0, 1.05), xlabel="Stock price S", ylabel="Digital price", title="Digital call")
ax[1].legend()
plt.tight_layout()
plt.show()

# Max errors on visible plot range
print("Max error on plotted range:")
print("European:", np.max(np.abs(V_eur_heat[range_eur] - V_eur_closed[range_eur])))
print("Digital: ", np.max(np.abs(V_dig_heat[range_dig] - V_dig_closed[range_dig])))
