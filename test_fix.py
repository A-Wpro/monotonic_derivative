from monotonic_derivative.monotonic_derivative import (
    ensure_monotonic_derivative,
)
from monotonic_derivative.curve_smoothing import curve_smoothing
import numpy as np

y = np.array([1, 1, 3.5, 5, 6, 6.1, 4, 5])
x = np.arange(len(y))


y_ensure = ensure_monotonic_derivative(
    x=np.arange(len(y)),
    y=y,
    degree=2,
    force_negative_derivative=True,
    verbose=True,
    save_plot=True,
)
print(y_ensure)
y_smooth = curve_smoothing(
    y_ensure, population_size=100, num_generations=1000, alpha=0.5, save_plots=True
)
print(y_smooth)
