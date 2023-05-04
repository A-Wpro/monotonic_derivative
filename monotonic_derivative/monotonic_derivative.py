import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import imageio


def ensure_monotonic_derivative(
    x, y, degree=2, force_negative_derivative=False, verbose=False, save_plot=False
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

    Returns:
    modified_y: numpy array, the modified dependent variable data points
    """

    def objective_function(y, x, y_original):
        return np.sum((y - y_original) ** 2)

    def constraint_second_derivative_increasing(y, x):
        """
        Function to create a constraint function to be used in the optimization problem.
        The constraint function calculates the minimum difference between consecutive second derivatives.

        Parameters:
        y: numpy array, the dependent variable data points
        x: numpy array, the independent variable data points

        Returns:
        constraint: function, the constraint function for the optimization problem
        """

        def constraint(y):
            cs = CubicSpline(x, y)
            y_2nd_derivative = cs(x, 2)
            return np.min(np.diff(y_2nd_derivative))

        return constraint

    def constraint_second_derivative_decreasing(y, x):
        """
        Function to create a constraint function to be used in the optimization problem.
        The constraint function calculates the maximum difference between consecutive second derivatives.

        Parameters:
        y: numpy array, the dependent variable data points
        x: numpy array, the independent variable data points

        Returns:
        constraint: function, the constraint function for the optimization problem
        """

        def constraint(y):
            cs = CubicSpline(x, y)
            y_2nd_derivative = cs(x, 2)
            return np.min(-np.diff(y_2nd_derivative))

        return constraint

    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    # Check if the specified degree is valid
    if degree >= len(y) - 1:
        raise ValueError(
            "Degree must be less than the length of the data minus 1, since we lose one data point for each derivative and need at least two data points"
        )

    # Define the constraint for the optimization problem
    if force_negative_derivative:
        cons = {"type": "ineq", "fun": constraint_second_derivative_decreasing(y, x)}
    else:
        cons = {"type": "ineq", "fun": constraint_second_derivative_increasing(y, x)}

    # Solve the optimization problem
    res = minimize(objective_function, y, args=(x, y), constraints=cons)
    modified_y = res.x

    if verbose:
        print("Original y    :", y)
        print("Modified y    :", modified_y)
        print("Optimization success:", res.success)
        print("Optimization message:", res.message)

    # Save and display plots of original and modified data and their derivatives if save_plot is True
    if save_plot:
        fig, ax = plt.subplots(degree + 1, 1, figsize=(8, 3 * (degree + 1)))

        # Plot the original and modified data points
        ax[0].plot(x, y, "o--", label="Original data points")
        ax[0].plot(x, modified_y, "o--", label="Modified data points")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].legend()

        cs_original = CubicSpline(x, y)
        cs_modified = CubicSpline(x, modified_y)

        # Plot derivatives from 1st to the specified degree
        for d, ax_i in enumerate(ax[1:], start=1):
            ax_i.plot(
                x[:-d],
                np.diff(y, n=d) / np.prod([np.diff(x) for _ in range(d)]),
                "o--",
                label=f"{d}th derivative (original)",
            )
            ax_i.plot(
                x[:-d],
                np.diff(modified_y, n=d) / np.prod([np.diff(x) for _ in range(d)]),
                "o--",
                label=f"{d}th derivative (modified)",
            )
            ax_i.set_xlabel("x")
            ax_i.set_ylabel(f"{d}th derivative")
            ax_i.legend()

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = imageio.imread(buf)
        plt.close(fig)

        # Save the image as a png
        imageio.imwrite("derivative.png", image)

    return modified_y
