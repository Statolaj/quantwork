# Init
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import shap
from math import erf

# Seeds
np.random.seed(4321)
torch.manual_seed(4321)

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
def digital_formula(S, K, T, r, sigma):

    S = np.asarray(S, dtype=float)
    d2 = np.zeros_like(S)
    price = np.zeros_like(S)

    d2[S>0] = (np.log(S[S>0] / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    price[S>0] = np.exp(-r * T) * normal_cdf(d2[S>0])

    return price

# Neural network of two layers with 16 hidden units and softplus activation.
class BlackScholesNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.Softplus(),

            torch.nn.Linear(64, 64),
            torch.nn.Softplus(),

            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    
# Function to compute the predicted price
def predict_price(network, X_raw, y_train_raw):

    X_scaled = (X_raw - X_train_raw.mean(axis=0)) / X_train_raw.std(axis=0)
    X_torch = torch.tensor(X_scaled, dtype=torch.float32)

    network.eval()
    with torch.no_grad():
        y_scaled = network(X_torch).numpy().ravel()

    return y_train_raw.mean() + y_train_raw.std() * y_scaled

# Train the network
def train_networks(
    X_train_torch,
    y_european_train_torch,
    y_digital_train_torch,
):
    european_network = BlackScholesNet()
    digital_network = BlackScholesNet()

    european_optimizer = torch.optim.Adam(european_network.parameters(), lr=0.001)
    digital_optimizer = torch.optim.Adam(digital_network.parameters(), lr=0.001)

    loss_fn = torch.nn.MSELoss()

    n_steps = 1500

    for step in range(n_steps):

        european_optimizer.zero_grad()
        european_loss = loss_fn(european_network(X_train_torch), y_european_train_torch)
        european_loss.backward()
        european_optimizer.step()

        digital_optimizer.zero_grad()
        digital_loss = loss_fn(digital_network(X_train_torch), y_digital_train_torch)
        digital_loss.backward()
        digital_optimizer.step()

    return european_network, digital_network

#######################################################################

if __name__ == "__main__":

    # Generate data
    n_samples = 10000

    S0 = np.random.uniform(100, 175, n_samples)
    K = S0 + np.random.uniform(5, 50, n_samples)
    T = np.random.uniform(0.5, 2, n_samples)
    r = np.random.uniform(0, 0.1, n_samples)
    sigma = np.random.uniform(0.1, 0.9, n_samples)

    X = np.column_stack([S0, K, T, r, sigma])
    y_european = european_formula(*X.T)
    y_digital = digital_formula(*X.T)

    # Split into training and test part
    indices = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)

    train_id = indices[:n_train]
    test_id = indices[n_train:]

    X_train_raw = X[train_id]
    y_european_train_raw = y_european[train_id]
    y_digital_train_raw = y_digital[train_id]

    X_test_raw = X[test_id]
    y_european_test_raw = y_european[test_id]
    y_digital_test_raw = y_digital[test_id]

    X_train = (X_train_raw - X_train_raw.mean(axis=0)) / X_train_raw.std(axis=0)
    X_test = (X_test_raw - X_train_raw.mean(axis=0)) / X_train_raw.std(axis=0)

    y_european_train = (y_european_train_raw - y_european_train_raw.mean()) / y_european_train_raw.std()
    y_european_test = (y_european_test_raw - y_european_train_raw.mean()) / y_european_train_raw.std()

    y_digital_train = (y_digital_train_raw - y_digital_train_raw.mean()) / y_digital_train_raw.std()
    y_digital_test = (y_digital_test_raw - y_digital_train_raw.mean()) / y_digital_train_raw.std()

    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_european_train_torch = torch.tensor(y_european_train.reshape(-1, 1), dtype=torch.float32)
    y_digital_train_torch = torch.tensor(y_digital_train.reshape(-1, 1), dtype=torch.float32)

    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    # Training the network
    european_network, digital_network = train_networks(X_train_torch, y_european_train_torch, y_digital_train_torch)

    ############################################################################

    print("European call option:")

    # Evaluate the sup test error and 95% error
    print("0.95 error:  ", np.percentile(np.abs(predict_price(european_network, X_test_raw, y_european_train_raw) - y_european_test_raw), 95))
    print("Max error:  ", np.max(np.abs(predict_price(european_network, X_test_raw, y_european_train_raw) - y_european_test_raw)))

    # Plot predicted vs true prices
    plt.figure(figsize=(8, 8))
    plt.scatter(predict_price(european_network, X_test_raw, y_european_train_raw), y_european_test_raw, alpha=0.5)
    plt.plot([0, 50], [0, 50], 'r--')
    plt.xlabel("Predicted Price")
    plt.ylabel("True Price")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.grid()
    plt.show()

    # Shap values
    shap_values = shap.KernelExplainer(lambda X_raw: predict_price(european_network, X_raw, y_european_train_raw),X_train_raw[:50]).shap_values(X_test_raw[:100], nsamples=100)
    shap_values = np.squeeze(np.asarray(shap_values))

    # S0
    S0_values = X_test_raw[:100, 0]
    S0_shap = shap_values[:, 0]
    S0_corr = np.corrcoef(S0_values, S0_shap)[0, 1]
    print(f"S0    : mean SHAP = {np.mean(S0_shap): .4f}, correlation = {S0_corr: .4f}")

    # K
    K_values = X_test_raw[:100, 1]
    K_shap = shap_values[:, 1]
    K_corr = np.corrcoef(K_values, K_shap)[0, 1]
    print(f"K     : mean SHAP = {np.mean(K_shap): .4f}, correlation = {K_corr: .4f}")

    # sigma
    sigma_values = X_test_raw[:100, 4]
    sigma_shap = shap_values[:, 4]
    sigma_corr = np.corrcoef(sigma_values, sigma_shap)[0, 1]
    print(f"sigma : mean SHAP = {np.mean(sigma_shap): .4f}, correlation = {sigma_corr: .4f}")

    ####################
    ####################
    ####################

    print("Digital call option:")

    # Evaluate the sup test error and 95% error
    print("0.95 error:  ", np.percentile(np.abs(predict_price(digital_network, X_test_raw, y_digital_train_raw) - y_digital_test_raw), 95))
    print("Max error:  ", np.max(np.abs(predict_price(digital_network, X_test_raw, y_digital_train_raw) - y_digital_test_raw)))

    # Plot predicted vs true prices
    plt.figure(figsize=(8, 8))
    plt.scatter(predict_price(digital_network, X_test_raw, y_digital_train_raw), y_digital_test_raw, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("Predicted Price")
    plt.ylabel("True Price")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

    # Shap values
    shap_values = shap.KernelExplainer(lambda X_raw: predict_price(digital_network, X_raw, y_digital_train_raw),X_train_raw[:50]).shap_values(X_test_raw[:100], nsamples=100)
    shap_values = np.squeeze(np.asarray(shap_values))

    # S0
    S0_values = X_test_raw[:100, 0]
    S0_shap = shap_values[:, 0]
    S0_corr = np.corrcoef(S0_values, S0_shap)[0, 1]
    print(f"S0    : mean SHAP = {np.mean(S0_shap): .4f}, correlation = {S0_corr: .4f}")

    # K
    K_values = X_test_raw[:100, 1]
    K_shap = shap_values[:, 1]
    K_corr = np.corrcoef(K_values, K_shap)[0, 1]
    print(f"K     : mean SHAP = {np.mean(K_shap): .4f}, correlation = {K_corr: .4f}")

    # sigma
    sigma_values = X_test_raw[:100, 4]
    sigma_shap = shap_values[:, 4]
    sigma_corr = np.corrcoef(sigma_values, sigma_shap)[0, 1]
    print(f"sigma : mean SHAP = {np.mean(sigma_shap): .4f}, correlation = {sigma_corr: .4f}")
