import numpy as np
import cvxpy as cp

def run(input_data, solver_params=None, extra_arguments=None):
    """
    Solves the index tracking optimization problem.

    Parameters:
    - input_data: dict, containing 'returns_matrix' and 'index_returns'.
    - solver_params: dict, optional solver parameters (e.g., {"solver": "OSQP"}).
    - extra_arguments: dict, optional additional arguments (not used in this implementation).

    Returns:
    - dict: Contains 'optimum' (optimal objective value) and 'optimizer_vector' (optimal weights).
    """
    # Extract asset returns matrix and index returns vector
    returns_matrix = np.array(input_data.get("returns_matrix"))
    index_returns = np.array(input_data.get("index_returns"))

    # Get dimensions: T time periods, n assets
    T, n = returns_matrix.shape

    # Ensure that index_returns is a one-dimensional vector of length T
    if index_returns.ndim == 2 and index_returns.shape[1] == 1:
        index_returns = index_returns.flatten()
    elif index_returns.ndim != 1:
        raise ValueError("index_returns must be a one-dimensional array or a column vector.")

    # Define the optimization variable (portfolio weights, vector of length n)
    w = cp.Variable(n)

    # Define the objective: minimize the squared error between portfolio returns (R*w) and index returns.
    objective = cp.Minimize(cp.sum_squares(returns_matrix @ w - index_returns))

    # Define the constraints:
    constraints = [
        cp.sum(w) == 1,  # Full investment (weights sum to one)
        w >= 0           # No short-selling constraint.
    ]

    # Formulate and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(**solver_params)

    # Retrieve the optimal objective value and the optimizer vector.
    optimum = problem.value
    optimizer_vector = w.value.tolist()

    return {
        "optimum": optimum,
        "optimizer_vector": optimizer_vector
    }