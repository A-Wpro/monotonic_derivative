import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calculate_similarity(curve1, curve2):
    # Interpolate curve2 to have the same length as curve1
    f = interp1d(np.arange(len(curve2)), curve2)
    curve2_interp = f(np.linspace(0, len(curve2) - 1, len(curve1)))

    # Calculate correlation coefficient
    correlation = np.corrcoef(curve1, curve2_interp)[0, 1]

    # Convert correlation coefficient to similarity score
    similarity = (correlation + 1) / 2

    return similarity

def plot_derivatives(x, y_original, y_mod, max_order, flip=False, show=False):
    # Initialize the figure
    fig, axs = plt.subplots(max_order + 1, 1, figsize=(10, (max_order + 1) * 5))

    # Calculate and plot the original and smoothed curves
    cs_original = CubicSpline(x, y_original)
    cs_mod = CubicSpline(x, y_mod)
    x_interp = np.arange(np.min(x), np.max(x), 0.01)
    y_interp = cs_mod(x_interp)

    axs[0].plot(x, y_original, "ro-", alpha=1, label="Original curve", linewidth=1)
    axs[0].plot(x, y_mod, "o--", alpha=0.6, label="cubic curve", linewidth=2)
    axs[0].legend()

    # Calculate and plot each order of derivative
    for order in range(1, max_order + 1):
        y_original_derivative = cs_original(x, order, extrapolate=False)
        y_derivative = cs_mod(x, order)
        y_interp_derivative = cs_mod(x_interp, order)

        if flip:
            y_original_derivative = -y_original_derivative
            y_derivative = -y_derivative
            y_interp_derivative = -y_interp_derivative

        axs[order].plot(
            x,
            y_original_derivative,
            "g",
            label=f"Original Order {order} Derivative",
            linewidth=2,
            alpha=0.75,
        )
        axs[order].plot(
            x_interp,
            y_interp_derivative,
            "r",
            label=f"Smoothed Interp Order {order} Derivative",
            linewidth=2,
            alpha=0.5,
        )
        axs[order].plot(
            x,
            y_derivative,
            "b-",
            label=f"Smoothed Order {order} Derivative",
            linewidth=2,
            alpha=1,
        )
        axs[order].legend()

    # Set common x-axis label for all subplots
    fig.text(0.5, 0.04, "x", ha="center", fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Save the figure as an image file
    plt.savefig("derivative.png")

    if show:  # Display the figure
        plt.show()


def ensure_monotonic_derivative(
    x,
    y,
    degree=2,
    force_negative_derivative=False,
    verbose=False,
    save_plot=False,
    max_iter_minimize=50000,
):
    """
    Modify the given data points to ensure that the specified degree derivative of the cubic spline is always monotonically increasing or decreasing.

    Parameters:
    x: numpy array, the independent variable data points
    y: numpy array, the dependent variable data points
    degree: int, the degree of the derivative to check for monotonicity
    force_negative_derivative: bool, force the specified degree derivative to be monotonically decreasing if True
    verbose: bool, print additional information if True
    save_plot: bool, save the plots as GIF images if True
    max_iter_minimize: int, maximum number of iterations for the optimization method

    Returns:
    modified_y: numpy array, the modified dependent variable data points
    """

    def objective_function(
        y, x, y_original, degree
    ):  # parameter will be used to modify penalty if needed
        penalty = 0  # no penalty, it's here for potential futur update
        mse = np.sum((y - y_original) ** 2)
        # We add the penalty term to the MSE
        return mse + penalty

    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # Check if the specified degree is valid
    if degree >= len(y) - 1:
        raise ValueError(
            "Degree must be less than the length of the data minus 1, since we lose one data point for each derivative and need at least two data points"
        )
    if degree < 0:
        raise ValueError("The degree parameter cannot be negative")

    def constraint_function(y, x, degree, is_negative=False):
        """
        Function to create a constraint function to be used in the optimization problem.
        The constraint function checks if all points of the `degree`-th derivative are less than 0.

        Parameters:
        y: numpy array, the dependent variable data points
        x: numpy array, the independent variable data points
        degree: int, the degree of the derivative to check for negativity

        Returns:
        constraint: function, the constraint function for the optimization problem
        """

        def get_resampled_derivative(interpolator):
            x_resampled = np.arange(np.min(x), np.max(x), 0.01)
            derivative_resampled = interpolator(x_resampled, degree)
            return (
                derivative_resampled - 0.00000000001
            )  # To avoid critic state problem and other calculation problem

        def constraint(y):
            cs = CubicSpline(x, y)
            derivative_resampled = get_resampled_derivative(cs)
            if is_negative:
                return -derivative_resampled
            else:
                return derivative_resampled

        return constraint

    # Define the constraint for the optimization problem
    cons = {
        "type": "ineq",
        "fun": constraint_function(y, x, degree, force_negative_derivative),
    }

    # List of methods to try
    methods = [
        "SLSQP",
        "CG",
        "BFGS",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "trust-constr",
    ]

    # Solve the optimization problem
    res = None
    for method in methods:
        res = minimize(
            objective_function,
            y,
            args=(x, y, degree),
            constraints=cons,
            method=method,
            options={"maxiter": max_iter_minimize},
        )
        # if optimization is successful and the termination condition is not 'xtol', break the loop
        if res.success and "xtol" not in res.message:
            break

    modified_y = res.x if res else y

    if verbose:
        similarity_score = calculate_similarity(modified_y, y)
        print("Original y    :", y)
        print("Modified y    :", modified_y)
        print("Similarity score :", similarity_score)

        print("Optimization success:", res.success)
        print("Optimization message:", res.message)

    # Save and display plots of original and modified data and their derivatives if save_plot is True
    if save_plot:
        # Use the function
        plot_derivatives(x, y, modified_y, degree, flip=True, show=True)

    return modified_y
