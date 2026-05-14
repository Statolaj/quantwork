import contextlib
import io
from math import erf, exp, sqrt
import numpy as np
import torch

print("Importing Monte Carlo module...")
from bs_pricing_from_monte_carlo import monte_carlo_black_scholes
print("Importing Heat Equation module...")
from bs_pricing_from_heat_equation import solve_black_scholes_from_heat
print("Importing Neural Network module...")
from bs_pricing_from_neural_network import normal_cdf, european_formula, digital_formula, BlackScholesNet, predict_price, train_networks

# Numerical choices for the comparison.
N_space = 1001
N_paths = 1000000
N_nn = 20000

# ---------------------------------------------------------------------
# Neural-network setup.
# ---------------------------------------------------------------------


def train_neural_networks():
    np.random.seed(4321)
    torch.manual_seed(4321)

    # Similar training region as in bs_pricing_from_neural_network.py.
    S0_train = np.random.uniform(1, 105, N_nn)
    K_train = np.random.uniform(1, 105, N_nn)
    T_train = np.random.uniform(0.45, 1.1, N_nn)
    r_train = np.random.uniform(0, 0.1, N_nn)
    sigma_train = np.random.uniform(0.08, 0.82, N_nn)

    X = np.column_stack([S0_train, K_train, T_train, r_train, sigma_train])
    y_european = european_formula(*X.T)
    y_digital = digital_formula(*X.T)

    indices = np.random.permutation(N_nn)
    n_train = int(0.8 * N_nn)
    train_id = indices[:n_train]

    X_train_raw = X[train_id]
    y_european_train_raw = y_european[train_id]
    y_digital_train_raw = y_digital[train_id]

    X_train = (X_train_raw - X_train_raw.mean(axis=0)) / X_train_raw.std(axis=0)
    y_european_train = (y_european_train_raw - y_european_train_raw.mean()) / y_european_train_raw.std()
    y_digital_train = (y_digital_train_raw - y_digital_train_raw.mean()) / y_digital_train_raw.std()

    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    train_networks.__globals__["X_train_torch"] = X_train_torch
    predict_price.__globals__["X_train_raw"] = X_train_raw

    y_european_train_torch = torch.tensor(y_european_train.reshape(-1, 1), dtype=torch.float32)
    y_digital_train_torch = torch.tensor(y_digital_train.reshape(-1, 1), dtype=torch.float32)

    european_network, digital_network = train_networks(X_train_torch, y_european_train_torch, y_digital_train_torch)

    return european_network, digital_network, y_european_train_raw, y_digital_train_raw


def neural_prices(network, S0, K, T, r, sigma):
    european_network, digital_network, y_european_train_raw, y_digital_train_raw = network

    X_raw = np.array([[S0, K, T, r, sigma]], dtype=float)
    european_nn = predict_price(european_network, X_raw, y_european_train_raw)[0]
    digital_nn = predict_price(digital_network, X_raw, y_digital_train_raw)[0]

    return european_nn, digital_nn


# ---------------------------------------------------------------------
# Main comparison.
# ---------------------------------------------------------------------

def read_parameters():
    raw = input("Enter S0 K T r sigma, separated by spaces: ")
    values = raw.split()

    if len(values) != 5:
        raise ValueError("Please enter exactly five values: S0 K T r sigma")

    S0, K, T, r, sigma = map(float, values)

    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    return S0, K, T, r, sigma


def main():
    S0, K, T, r, sigma = read_parameters()

    closed_european = float(european_formula(np.array([S0]), K, T, r, sigma)[0])
    closed_digital = float(digital_formula(np.array([S0]), K, T, r, sigma)[0])

    # Heat equation method. Suppress diagnostic prints from the source function.
    with contextlib.redirect_stdout(io.StringIO()):
        heat_european = solve_black_scholes_from_heat(S0, K, T, r, sigma, "european", N_space)[0]
        heat_digital = solve_black_scholes_from_heat(S0, K, T, r, sigma, "digital", N_space)[0]

    # Monte Carlo method with antithetic variates.
    np.random.seed(4321)
    mc_european, mc_european_se, mc_digital, mc_digital_se = monte_carlo_black_scholes(
        S0, K, T, r, sigma, n_paths=N_paths
    )

    # Neural-network method.
    nn_european, nn_digital = neural_prices(network, S0, K, T, r, sigma)

    rows = [
        ("Closed form", closed_european, 0.0, closed_digital, 0.0),
        ("Heat PDE", heat_european, abs(heat_european - closed_european), heat_digital, abs(heat_digital - closed_digital)),
        ("Monte Carlo", mc_european, abs(mc_european - closed_european), mc_digital, abs(mc_digital - closed_digital)),
        ("Neural net", nn_european, abs(nn_european - closed_european), nn_digital, abs(nn_digital - closed_digital)),
    ]
    print("\nInput parameters")
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")

    print("\nPricing comparison")
    print(f"{'Method':<14}{'European':>14}{'Eur abs err':>14}{'Digital':>14}{'Dig abs err':>14}")
    print("-" * 75)
    for method, eur, eur_err, dig, dig_err in rows:
        print(f"{method:<14}{eur:>14.6f}{eur_err:>14.6f}{dig:>14.6f}{dig_err:>14.6f}")

# Run the main function when the script is executed
if __name__ == "__main__":
    print("Training neural networks...")
    network = train_neural_networks()
    print("Finished training neural networks. \n")
    while True:
        try:
            main()
            run_again = input("Run again with different parameters? (y/n): ")
            if run_again.lower() != 'y':
                break 
        except ValueError as e:
            print(f"Error: {e}. Please try again.\n")
