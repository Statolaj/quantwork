import random
import math
import pandas as pd

#########################################################################

# A) Normal Monte Carlo from N(0,1)
def Moment_A(n, m):
    values = [random.gauss(0.0, 1.0)**m for _ in range(n)]
    return sum(values) / n


# B) Importance sampling: sample from Normal with higher variance
def Moment_B(n, m):
    tau = m + 1    # variance

    incorrect_values = [random.gauss(0.0, math.sqrt(tau)) for _ in range(n)]
    adjusted_values = [x**m * math.sqrt(tau) * math.exp(-x*x/2 + x*x/(2*tau)) for x in incorrect_values]
    return sum(adjusted_values) / n

# C) Same importance sampling, but use antithetic variables for odd moments
def Moment_C(n, m):
    tau = m + 1  # variance

    # For odd moments, use antithetic pairing
    if m % 2 == 1:
        total = 0.0
        half_n = n // 2

        draws = [random.gauss(0.0, math.sqrt(tau)) for _ in range(half_n)]
        values = [(x**m + (-x)**m) * math.sqrt(tau) * math.exp(-x*x/2 + x*x/(2*tau)) for x in draws]
       
        return sum(values) / half_n

    # For even moments, same as Moment_B
    else:
        return Moment_B(n, m)
    
# D) Keep antithetic variables for odd moments, but use Chi-squared distribution for even moments with proposals
def Moment_D(n, m):

    # Odd moments
    if m % 2 == 1:
        return Moment_C(n, m)
    # Even moments
    else:
        k = m // 2
        alpha = k + 0.5   # shape
        theta = 2.0       # scale

        value = 0.0
        for _ in range(n):
            y = random.gammavariate(alpha, theta)

            f = (1 / math.sqrt(2 * math.pi * y)) * math.exp(-y / 2)
            q = (y**(alpha - 1) * math.exp(-y / theta)) / (math.gamma(alpha) * theta**alpha)

            value += (y**k) * f / q

        return value / n
    
# E) Symmetric two-Gaussian mixture for even moments plus antithetic variables for odd moments. Centers at $\pm sqrt(m)$ to capture the main contributions to the m-th moment of N(0,1).

def normal_density(x, mu, tau):
    return math.exp(-(x - mu)**2 / (2 * tau)) / math.sqrt(2 * math.pi * tau)

def Moment_E(n, m):

    # Odd moments
    if m % 2 == 1:
        return Moment_C(n, m)

    # Even moments:
    mu = math.sqrt(m)   # centers near the two important peaks ±sqrt(m)
    tau = 0.5          # local-curvature-matching default
    sigma = math.sqrt(tau)

    total = 0.0

    for _ in range(n):
        if random.random() < 0.5:
            x = random.gauss(mu, sigma)
        else:
            x = random.gauss(-mu, sigma)

        p = math.exp(-x*x / 2) / math.sqrt(2 * math.pi)

        q = 0.5 * normal_density(x, mu, tau) + 0.5 * normal_density(x, -mu, tau) # Proposal density q = 0.5*N(mu,tau) + 0.5*N(-mu,tau)

        total += x**m * p / q

    return total / n


# Test cases 
moments = [5, 10, 15, 20, 25, 30, 35, 40]
n_draws = 1000
n_replications = 1000

estimators = {
    "Moment_A": Moment_A,
    "Moment_B": Moment_B,
    "Moment_C": Moment_C,
    "Moment_D": Moment_D,
    "Moment_E": Moment_E,
}

# Estimate variance of MC estimators 
results = {}
all_estimates = {}

for name, estimator in estimators.items():
    row = []
    est_dict = {}

    for m in moments:
        estimates = [estimator(n_draws, m) for _ in range(n_replications)]
        est_dict[m] = estimates

        mean_est = sum(estimates) / n_replications
        var_est = sum((x - mean_est)**2 for x in estimates) / (n_replications - 1)
        row.append(var_est)

    results[name] = row
    all_estimates[name] = est_dict

# First table 
df = pd.DataFrame(results, index=[f"m={m}" for m in moments]).T
print("Variance table:")
print(df)

#Theoretical standard normal moment
def theoretical_moment(m):
    if m % 2 == 1:
        return 0.0
    val = 1
    for j in range(1, m, 2):
        val *= j
    return float(val)

# Table 2: theoretical vs Moment_D 
theoretical_vals = [theoretical_moment(m) for m in moments]
moment_d_vals = [sum(all_estimates["Moment_D"][m]) / n_replications for m in moments]

df2 = pd.DataFrame(
    {
        f"m={m}": [theoretical_vals[i], moment_d_vals[i]]
        for i, m in enumerate(moments)
    },
    index=["Theoretical", "Moment_D"]
)

print("Exact vs Moment_D:")
print(df2)
