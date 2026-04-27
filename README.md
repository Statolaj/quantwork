# quantwork
A repository dedicated containing simple programming tasks/solutions relevant for me in computational finance.

The current list of files includes:
  1) $\mathtt{dice_game.py}$ which implements an optimal strategy for stopping in a dice game where you roll a six-sided die for a fixed number of rounds. The goal is maximizing the expected number of pips. Uses recursion.
  2) $\mathtt{gaussian_moments.py}$ which simulates the standard Gaussian moments using simple Monte Carlo. The files also contains improvements with a final version using anti-thetic variables for odd moments and the fact that squaring standard gaussian variable yields the $\chi^2$ distribution for even moments.
  3) $\mathtt{heat_equation_solver.py}$ which implements a finite difference method for solving the heat equation with a boundary condition in dimensions 1 and 2. The method can be extended to higher dimension, but due to the fact that the number of grid points grows exponentially in the dimension, it is not possible to apply this method in practice. Instead use Feynmann-Kac to and simulate the expectation using Monte Carlo, where the precision does not depend on the dimension.  
