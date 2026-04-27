# Init
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to put all boundary values equal to 0.
def boundary_zero(A):

    d = A.ndim

    for j in range(d):
        left_boundary = [slice(None)] * d
        right_boundary = [slice(None)] * d

        left_boundary[j] = 0
        right_boundary[j] = -1

        A[tuple(left_boundary)] = 0.0
        A[tuple(right_boundary)] = 0.0

    return A

# Function that computes the Laplacian in every interior grid point using finite differences as approximation of second derivative. Delta is the grid spacing.
def laplacian(A, delta):

    d = A.ndim
    index = (slice(1, -1),) * d
    second_derivatives = []

    for j in range(d):
        front, back = list(index), list(index)
        front[j] = slice(2, None) # Move forward in the j-th direction
        back[j] = slice(None, -2) # Move backward in the j-th direction

        # Central difference approximation of the second derivative in the j-th direction
        second_derivative_value = (A[tuple(front)]- 2 * A[tuple(index)] + A[tuple(back)]) / delta**2
        second_derivatives.append(second_derivative_value)

    laplacian_interior = sum(second_derivatives) 
    B = np.zeros_like(A)
    B[index] = laplacian_interior

    return B

# The main function. Takes dimension, number of grid points per dimension, terminal time, diffusion coefficient and initial condition as input and returns the numerical solution at terminal time on the grid.
def solve_heat_equation(d, n_divisions, T, alpha, initial_condition):

    # Grid here
    x = np.linspace(0, 1, n_divisions)
    grid = np.meshgrid(*([x] * d), indexing="ij")
    delta = x[1] - x[0]

    # Stability condition for the explicit finite difference method
    delta_t = 0.45 * delta**2 / (2 * d * alpha)
    n_steps = int(np.ceil(T / delta_t))
    delta_t = T / n_steps # So delta_t divides T exactly

    print(f"Dimension d = {d}"), print(f"delta = {delta}"), print(f"delta_t = {delta_t}"), 
    print(f"Number of time steps = {n_steps}"), print(f"Stability ratio = {alpha * delta_t / delta**2}"), print(f"Stability bound = {1 / (2 * d)}")

    # Initial condition and zeros
    A = initial_condition(*grid)
    A = boundary_zero(A)

    # Algorithm running the finite difference method for n_steps time steps
    for step in range(n_steps):
        A = A + delta_t * alpha * laplacian(A, delta)
        A = boundary_zero(A)

    return A, grid

# 1D test case using x \mapsto sin(pi x) as initial condition, which has solution: u(t,x) = exp(-pi^2 t) sin(pi x).
def initial_condition_1(x):
    return np.sin(np.pi * x)

start_time = time.perf_counter()

A, grid = solve_heat_equation(d=1,n_divisions=100,T=0.1,alpha=1,initial_condition=initial_condition_1)
x = grid[0]
A_teori = np.exp(-np.pi**2 * 0.1) * np.sin(np.pi * x)

end_time = time.perf_counter()

print("Running time:", end_time - start_time, "seconds")
print("Max error:", np.max(np.abs(A - A_teori)))

plt.plot(x, A, label="Numerical")
plt.plot(x, A_teori, "--", label="Theory")
plt.legend(), plt.xlabel("x"), plt.ylabel("u(T,x)"), plt.title("1D heat equation")
plt.show()


# 2D test case using u(0,x,y) = sin(pi x) sin(pi y) with solution: u(t,x,y) = exp(-2 pi^2 t) sin(pi x) sin(pi y)
def initial_condition_2(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


start_time = time.perf_counter()

A, grid = solve_heat_equation(
    d=2,
    n_divisions=100,
    T=0.05,
    alpha=1,
    initial_condition=initial_condition_2
)

end_time = time.perf_counter()

x, y = grid

A_teori = np.exp(-2 * np.pi**2 * 0.05)* np.sin(np.pi * x)* np.sin(np.pi * y)

print("Running time:", end_time - start_time, "seconds")
print("Max error:", np.max(np.abs(A - A_teori)))

plt.imshow(A, origin="lower", extent=[0, 1, 0, 1])
plt.colorbar(label="u(T,x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D heat equation")
plt.show()