import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import imageio


def extend_data(x, y, dx=0.1, dy=0.1, force_negative_derivative=False):
    """
    Add a new data point before the first point in the dataset.

    Parameters:
    x: numpy array, the independent variable data points
    y: numpy array, the dependent variable data points
    dx: float, the distance between the new data point and the first data point in x
    dy: float, the distance between the new data point and the first data point in y
    force_negative_derivative: bool, force the specified degree derivative to be monotonically decreasing if True

    Returns:
    x_extended: numpy array, the extended x data points
    y_extended: numpy array, the extended y data points
    """
    x_extended = np.insert(x, 0, x[0] - dx)
    y_extended = np.insert(
        y, 0, y[0] - dy if not force_negative_derivative else y[0] + dy
    )

    return x_extended, y_extended


def ensure_monotonic_derivative(
    x,
    y,
    degree=2,
    force_negative_derivative=False,
    verbose=False,
    save_plot=False,
    use_interpolated_data=False,
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
    use_interpolated_data: bool, use interpolated data points instead of the original ones if True
    max_iter_minimize: int, maximum number of iterations for the optimization method
    num_points: int, the number of points to use for interpolation if use_interpolated_data is True

    Returns:
    modified_y: numpy array, the modified dependent variable data points
    """
    # Extend the data with a new point before the first point
    x, y = extend_data(
        x, y, dx=1, dy=0, force_negative_derivative=force_negative_derivative
    )

    def objective_function(y, x, y_original):
        # This is the original objective
        mse = np.sum((y - y_original) ** 2)

        # This is the penalty term. You can adjust the scale factor (e.g., 1e3)
        # to make the penalty larger or smaller.
        # first_point_penalty = 1e3 * max(0, 0.1 - abs(y[0] - y_original[0])) ** 4

        return mse  # + first_point_penalty

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

    def interpolate_data(x, y, num_points=1000):
        """
        Create a cubic spline interpolation of the given data points with the specified degree.

        Parameters:
        x: numpy array, the independent variable data points
        y: numpy array, the dependent variable data points
        num_points: int, the number of points to use for interpolation

        Returns:
        x_new: numpy array, the new x values used for interpolation
        y_new: numpy array, the new y values obtained from interpolation
        """
        cs = CubicSpline(x, y, bc_type="natural")

        x_new = np.linspace(x[0], x[-1], num_points)
        y_new = cs(x_new)

        return x_new, y_new

    # Use interpolated data if use_interpolated_data is True
    label_original_point = "Original data points"
    if use_interpolated_data:
        x, y = interpolate_data(x, y, len(y) + degree)
        label_original_point = "Original extrapoled data points"

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
    # List of methods to try
    methods = [
        "SLSQP",
        "CG",
        "BFGS",
        "Newton-CG",
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
            args=(x, y),
            constraints=cons,
            method=method,
            options={"maxiter": max_iter_minimize},
        )
        # if optimization is successful and the termination condition is not 'xtol', break the loop
        if res.success and "xtol" not in res.message:
            break
    y = y[1:]  # to remove 1st fake dot
    modified_y = res.x[1:] if res else y
    x = x[1:]
    if verbose:
        print("Original y    :", y)
        print("Modified y    :", modified_y)
        print("Optimization success:", res.success)
        print("Optimization message:", res.message)

    # Save and display plots of original and modified data and their derivatives if save_plot is True
    if save_plot:
        fig, ax = plt.subplots(degree + 1, 1, figsize=(8, 3 * (degree + 1)))

        # Plot the original and modified data points
        ax[0].plot(x, y, "o--", label=label_original_point)
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
