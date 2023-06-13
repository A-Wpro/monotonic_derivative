import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator


def extend_data(x, y, dx=0.1, dy=0.1, force_negative_derivative=False):
    """
    Add a new data point before the first point and after the last point in the dataset.

    Parameters:
    x: numpy array, the independent variable data points
    y: numpy array, the dependent variable data points
    dx: float, the distance between the new data point and the first/last data point in x
    dy: float, the distance between the new data point and the first/last data point in y
    force_negative_derivative: bool, force the specified degree derivative to be monotonically decreasing if True

    Returns:
    x_extended: numpy array, the extended x data points
    y_extended: numpy array, the extended y data points
    """
    x_extended = np.concatenate(([x[0] - dx], x, [x[-1] + dx]))
    y_extended = np.concatenate(
        (
            [y[0] * (1 - dy) if not force_negative_derivative else y[0] * (1 + dy)],
            y,
            [y[-1] * (1 - dy) if not force_negative_derivative else y[-1] * (1 + dy)],
        )
    )

    return x_extended, y_extended


def calculate_similarity(curve1, curve2):
    # Interpolate curve2 to have the same length as curve1
    f = interp1d(np.arange(len(curve2)), curve2)
    curve2_interp = f(np.linspace(0, len(curve2) - 1, len(curve1)))

    # Calculate correlation coefficient
    correlation = np.corrcoef(curve1, curve2_interp)[0, 1]

    # Convert correlation coefficient to similarity score
    similarity = (correlation + 1) / 2

    return similarity


def interpolated_curve(x, y, step=0.01):
    # Create a cubic spline interpolation
    interp = PchipInterpolator(x, y, extrapolate=True)

    # Control 2 decimals for GL
    x_pchip_original = np.arange(
        round(float(x.min()), 2),
        round(float(x.max()), 2) + step,
        step,
    )
    y_pchip_original = interp(x_pchip_original)

    return x_pchip_original, y_pchip_original


def ensure_monotonic_derivative(
    x,
    y,
    degree=2,
    force_negative_derivative=False,
    verbose=False,
    save_plot=False,
    max_iter_minimize=50000,
    return_interpolated_curve=False,
    use_interpolated_data=False,  # Should not be use, except you know what you do
    extending_data=False,  # Should not be use, except you know what you do
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
    return_interpolated_curve : bool, instead of return the modified y alone we return a tuple of interpolated array.
    use_interpolated_data: bool, use interpolated data points instead of the original ones if True # Should not be use, except you know what you do
    extending_data: bool, create new data point before our data and after  # Should not be use, except you know what you do

    Returns:
    modified_y: numpy array, the modified dependent variable data points
    """
    # Extend the data with a new point before the first point
    if extending_data:
        x, y = extend_data(
            x,
            y,
            dx=(sum(x) / len(x)),
            dy=0.05,
            force_negative_derivative=force_negative_derivative,
        )  ## TODO, add dy,dx as value that can change to force the monotinic value, bad idea ?

    def objective_function(
        y, x, y_original, force_negative_derivative, degree
    ):  # parameter will be used to modify penalty if needed
        penalty = 0  # no penalty, it's here for potential futur update
        mse = np.sum((y - y_original) ** 2)
        # We add the penalty term to the MSE
        return mse + penalty

    def constraint_degree_derivative_positive(y, x, degree):
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

        def constraint(y):
            cs = CubicSpline(x, y)
            x_resampled = np.arange(np.min(x), np.max(x), 0.01)
            derivative_resampled = cs(x_resampled, degree)
            return (
                derivative_resampled - 0.00000000001
            )  # 0.00000000001 instead of 0 to avoid critic state problem and other calculation problem

        return constraint

    def constraint_degree_derivative_negative(y, x, degree):
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

        def constraint(y):
            cs = CubicSpline(x, y)
            x_resampled = np.arange(np.min(x), np.max(x), 0.01)
            derivative_resampled = cs(x_resampled, degree)
            return (
                -derivative_resampled - 0.00000000001
            )  # 0.00000000001 instead of 0 to avoid critic state problem and other calculation problem

        return constraint

    def interpolate_data(
        x, y, num_points_between=1000
    ):  # should be change very soon and should not be used like this except you know what yo do
        """
        Create a cubic spline interpolation of the given data points with the specified degree.
        New points are added between original points, but original points are also kept.

        Parameters:
        x: numpy array, the independent variable data points
        y: numpy array, the dependent variable data points
        num_points_between: int, the number of points to interpolate between each original pair of points

        Returns:
        x_new: numpy array, the new x values used for interpolation
        y_new: numpy array, the new y values obtained from interpolation
        """

        x_new = []
        y_new = []

        # Iterate over pairs of points in the original data
        for i in range(len(x) - 1):
            # Create a cubic spline for this pair of points
            cs = CubicSpline(x[i : i + 2], y[i : i + 2], bc_type="natural")

            # Generate new x values between this pair of original points
            x_interp = np.linspace(x[i], x[i + 1], num_points_between + 2)

            # Generate new y values for these x values
            y_interp = cs(x_interp)

            # Append new values to the list, excluding the last point to avoid duplicates
            x_new.extend(x_interp[:-1])
            y_new.extend(y_interp[:-1])

        # Append the last point from the original data
        x_new.append(x[-1])
        y_new.append(y[-1])

        # Convert lists to numpy arrays
        x_new = np.array(x_new)
        y_new = np.array(y_new)

        return x_new, y_new

    # Use interpolated data if use_interpolated_data is True
    label_original_point = "Original data points"
    if use_interpolated_data:
        y_not_intra = y
        x_not_intra = x
        x, y = interpolate_data(x, y, len(y) + degree)
        label_original_point_intra = "Original intrapoled data points"

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

    # Define the constraint for the optimization problem
    if force_negative_derivative:
        cons = {
            "type": "ineq",
            "fun": constraint_degree_derivative_negative(y, x, degree),
        }
    else:
        cons = {
            "type": "ineq",
            "fun": constraint_degree_derivative_positive(y, x, degree),
        }

    # Solve the optimization problem
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
            args=(x, y, force_negative_derivative, degree),
            constraints=cons,
            method=method,
            options={"maxiter": max_iter_minimize},
        )
        # if optimization is successful and the termination condition is not 'xtol', break the loop
        if res.success and "xtol" not in res.message:
            break
    if extending_data:
        y = y[0:-1]  # to remove 1st fake dot and last
        modified_y = res.x[0:-1] if res else y
        x = x[0:-1]
    else:
        modified_y = res.x if res else y

    if verbose:
        if use_interpolated_data:
            print("Original y    :", y_not_intra)
            print("Original intrapoled y    :", y)
            print("Modified y    :", modified_y)
            similarity_score_intra = calculate_similarity(modified_y, y)
            similarity_score = calculate_similarity(modified_y, y_not_intra)
            print("Similarity score intrapoled:", similarity_score_intra)
            print("Similarity score :", similarity_score)
        else:
            similarity_score = calculate_similarity(modified_y, y)
            print("Original y    :", y)
            print("Modified y    :", modified_y)
            print("Similarity score :", similarity_score)

        print("Optimization success:", res.success)
        print("Optimization message:", res.message)

    # Save and display plots of original and modified data and their derivatives if save_plot is True
    if save_plot:
        fig, ax = plt.subplots(degree + 1, 1, figsize=(8, 3 * (degree + 1)))

        # Plot the original and modified data points

        if use_interpolated_data:
            ax[0].plot(x, y, "b-", label=label_original_point_intra)
            ax[0].plot(x_not_intra, y_not_intra, "m-", label=label_original_point)
        else:
            ax[0].plot(x, y, "b-", label=label_original_point)
        ax[0].plot(x, modified_y, "g-", label="Modified data points")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].legend()

        cs_original = CubicSpline(x, y)
        cs_modified = CubicSpline(x, modified_y)

        # Plot derivatives from 1st to the specified degree
        for d, ax_i in enumerate(ax[1:], start=1):
            # Compute the Xth derivative
            y_first_derivative = cs_original(x, d)
            y_smoothed_first_derivative = cs_modified(x, d)
            ax_i.plot(
                x,
                y_first_derivative,
                label=f"{d}th derivative (original)",
            )
            ax_i.plot(
                x,
                y_smoothed_first_derivative,
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

    if return_interpolated_curve:
        return interpolated_curve(x, modified_y)
    return modified_y
