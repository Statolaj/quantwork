# quantwork

A repository dedicated to simple programming tasks relevant to computational finance. In short:

1. The folder `black_scholes` contains several Python files related to pricing in the Black-Scholes model. These include pricing European and digital call options by:
   - numerically solving the pricing PDE,
   - training a two-layer neural network to learn the pricing formula from simulated training data,
   - Monte Carlo estimation of the risk-neutral valuation formula.

   A direct comparison of these methods can be found [here](https://github.com/Statolaj/quantwork/blob/main/black_scholes/master_pricing_comparison.py). Additionally, there is a script for recovering the implied volatility from an option price.

2. The folder `misc_scripts` contains several Python and C++ files implementing a dice game, a heat equation solver, Monte Carlo estimation of Gaussian moments, and Monte Carlo estimation of $\pi$ with and without parallel programming.

3. The folder `notes` contains some mathematical background and model specifications used in the Black-Scholes pricing methods.

# black_scholes

1. `bs_pricing_heat_equation.py` contains an extended heat equation solver compared to `heat_equation_solver.py`, which can handle boundary conditions other than 0. In particular, we implement the ideas in `bs_pricing_heat_equation.pdf` and numerically solve the pricing PDEs arising from European and digital call options.

   We place the grid such that $\log(K)$ is the center grid point for both options. The file contains a test case with $S_0 = 200$, $K = 220$, $T = 1$, $r = 0.05$, and $\sigma = 0.5$. The option price is plotted against the stock price on the grid. The deviation from the theoretical closed-form pricing formulas is less than $0.002$ for both option types, with closed-form prices $35.925$ and $0.349$ for the European and digital call options, respectively.

2. `bs_pricing_neural_network.py` contains a neural network approximation of the pricing formula for a European call option. In particular, the model, consisting of two hidden layers of width 64, is trained on a large number of labeled points to minimize the $L^2$-distance between test prices and predicted prices from the model.

   The 95th percentile of the absolute deviation is roughly 1.5. Finally, we look at SHAP values and recover financially intuitive relationships: decreasing strike $K$, increasing volatility $\sigma$, and increasing stock price $S_0$ all increase the option price in the model.

   A possible improvement would be to switch to a four-parameter version using $\log(S_0/K)$ instead of treating $S_0$ and $K$ as separate inputs.

3. `bs_pricing_monte_carlo.py` uses Monte Carlo estimation and antithetic variates to simulate the risk-neutral price of digital and European call options. The script runs quite fast.

4. `implied_volatility.py` implements both the bisection method and Newton's method to numerically solve for the value of $\sigma$ that yields a given option price in the Black-Scholes formula.

   Newton's method relies on vega and often converges faster. However, the test cases in the file illustrate that for very small initial guesses for $\sigma$, or for large strike prices $K$ where vega is close to 0, Newton's method can fail. On the other hand, bisection is slower but much more robust.

5. `master_pricing_comparison.py` imports modules from `bs_pricing_heat_equation.py`, `bs_pricing_neural_network.py`, and `bs_pricing_monte_carlo.py` to compare the resulting prices with the closed-form Black-Scholes formula. The user inputs the stock price, strike price, maturity, interest rate, and volatility.

   Initially training the neural network is the main bottleneck in terms of running time. From testing, it seems that solving the pricing PDE performs best, closely followed by Monte Carlo estimation, while the neural network performs somewhat worse.

# misc_scripts

1. `pi_estimation.cpp` implements a simple Monte Carlo estimation of $\pi$ in C++ using uniform samples from a $2 \times 2$ square and computing the fraction of points that land inside the unit circle.

   With uniform samples, the acceptance rate is $\pi/4 \approx 78.5\%$. One possible idea for improving this is to use importance sampling, for example by switching to a $2 \cdot \mathrm{Beta}(2,2)$ distribution, so that points are generally sampled closer to the center.

2. `parallel_pi_estimation.cpp` improves the Monte Carlo simulation of $\pi$ by using multiple threads. On my PC, using all 20 available threads, the program goes from running $n = 10^8$ samples in 20.3 seconds to 2.2 seconds, corresponding to roughly a factor 10 improvement.

3. `heat_equation_solver.py` implements a finite difference method for solving the heat equation with boundary conditions in dimensions 1 and 2.

   The method can be extended to higher dimensions, but since the number of grid points grows exponentially in the dimension, it is not practical in high dimensions. Instead, one can use the Feynman-Kac representation and simulate the expectation using Monte Carlo, where the convergence rate does not depend directly on the dimension.

4. `dice_game.py` implements an optimal stopping strategy for a dice game where one rolls a six-sided die for a fixed number of rounds. The goal is to maximize the expected number of pips. The solution uses recursion.

5. `gaussian_moments.py` simulates the $m$-th moment of the standard Gaussian distribution using Monte Carlo.

   The file contains five different versions:

   - A: Simple Monte Carlo.
   - B: Importance sampling with variance $m+1$.
   - C: Importance sampling with variance $m+1$ and antithetic variables.
   - D: Importance sampling using a $\Gamma((m+1)/2, 2)$ distribution and antithetic variables.
   - E: A mixture of two Gaussians centered at $\pm \sqrt{m}$ with antithetic variables.

   The best-performing version is D, where the estimator has essentially zero variance.

# notes

1. `bs_pricing_heat_equation.pdf` contains a short derivation of the transformation from the Black-Scholes partial differential equation to a heat equation, and explains how the solution of this equation can be used to price the original option.

   Emphasis is placed on the European call option $\Phi(S_T) = (S_T - K)^+$ and the digital call option $\Phi(S_T) = \mathbf{1}(S_T > K)$.

2. `bs_pricing_neural_network.pdf` contains the precise mathematical description of the neural network used in `bs_pricing_neural_network.py`. This includes a precise description of the softplus activation function and some description of the SHAP values.

3. `bs_pricing_monte_carlo.pdf` contains the precise formula used to simulate antithetic variates in `bs_pricing_monte_carlo.py`. In particular, since both the european call option and digital call option are non-decreasing in the stock price, we expect that antithetic variates yields a variance reduction of the estimator.
