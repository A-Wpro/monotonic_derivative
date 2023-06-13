## Monotonic Derivative - A Python Library

Monotonic Derivative is a Python library designed to modify real-life data to ensure that the specified degree derivative of the cubic spline is always monotonically increasing or decreasing. This library can be particularly useful in applications where the derivatives of the given data must follow specific monotonicity constraints, such as in scientific modeling or engineering applications.

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Real-life Applications](#real-life-applications)
- [Contributing](#contributing)
- [License](#license)

### Installation

To install the Monotonic Derivative library:

```
pip install monotonic-derivative
```

### Usage

#### Derivative tool:

First, import the `ensure_monotonic_derivative` function from the `monotonic_derivative` module:

```python
from monotonic_derivative import ensure_monotonic_derivative
```

The primary function of the library, `ensure_monotonic_derivative`, takes the following arguments:

- `x`: numpy array, the independent variable data points
- `y`: numpy array, the dependent variable data points
- `degree`: int, the degree of the derivative to check for monotonicity (default: 2)
- `force_negative_derivative`: bool, force the specified degree derivative to be monotonically decreasing if True (default: False)
- `verbose`: bool, print additional information if True (default: False)
- `save_plot`: bool, save the plots as PNG images if True (default: False)

The function returns a modified numpy array of dependent variable data points (`modified_y`) that satisfy the specified monotonicity constraints.

#### Example

```python
import numpy as np
from monotonic_derivative import ensure_monotonic_derivative

y = np.array([289.624, 430.174, 437.31, 434.67, 427.255, 408.902, 379.064])
x = np.array([0.078, 1.298, 2.791, 4.65, 6.002, 7.747, 10.0])

modified_y = ensure_monotonic_derivative(
    x=x,
    y=y,
    degree=3, 
    force_negative_derivative=False,
    verbose=True,
    save_plot=True,
)
```

#### Result :

We got a new curve that follow one rule : The Xth degree (depending of paramter degree) derivate must be increasing/decreasing (depending of force_negative_derivative parameter) curve.

```
Original y    : [289.624 430.174 437.31  434.67  427.255 408.902 379.064]
Modified y    : [290.6626736  423.94170784 445.4551795  434.69854619 424.27831859 407.55132434 380.41124237]
Similarity score : 0.9982442661935377 
Optimization success: True
Optimization message: Optimization terminated successfully
```

As you can see, big change on 3rd derivate have small impact of the 1st derivate curve but force it to be monotonicaly decreasing. 

![Derivative Example](./images/derivative.png)

Obvisouly and depending of which degree to lock below or above 0 you can force a curve to be closer to their mathematical reality.


#### Genetic algorithm smoothing tool:

```python
### Test for curve smoothing with genetic algo
from monotonic_derivative.curve_smoothing import  curve_smoothing
```

The primary function of the library, `curve_smoothing`, takes the following arguments:

- `points`: numpy array, an array of Y-coordinates representing the original curve
- `population_size`: int, the number of individuals in the population (default: 100)
- `num_generations`: int, the number of generations to run the algorithm (default: 1000)
- `mutation_rate`: float, the probability of a mutation occurring during reproduction (default: 0.1)
- `alpha`: float, the trade-off between smoothness and similarity to the original curve (default: 0.5)
- `save_plots`: bool, whether to save the progress of the algorithm as a GIF (default: False)
- `output_folder`: str, the folder to save the progress GIF if `save_plots` is True (default: "output")

The function returns the best individual (smoothed curve) found by the genetic algorithm as a numpy array.

#### Example

```python
import numpy as np
from monotonic_derivative.curve_smoothing import curve_smoothing

points = np.array([125, 55, 53, 40, 35, -25])
best_individual = curve_smoothing(points, alpha=0.5,save_plots = True)
#best_individual is the smoothed curve
```

#### Result :

This gif show how the genetic algorithm worked :

![Derivative Example](./images/progress.gif)

### Real-life Applications

In many real-life scenarios, the collected data may produce curves that are not logical or do not follow the expected constraints. For example, the data representing the velocity of a car over time should show an increasing or decreasing acceleration, but due to measurement errors or other factors, the collected data points may not reflect this.

The Monotonic Derivative library offers an easy solution to slightly modify the data to respect these constraints. By using this library, you can ensure that the specified degree derivative always monotonically increasing or decreasing, by forcing the value of a derivative to be below or above 0, making your data analysis more accurate and reliable.

### Limit

There are certain limitations to consider when using this library. For example, if you want to enforce a 2nd-degree monotonic increasing derivative on a curve that already has an originally monotonic increasing 2nd-degree derivative, the modification applied to the original curve will be much more pronounced. In this case, the resulting 2nd-degree derivative will be almost flat, but still slightly increasing.

It's important to remember that the purpose of this library is to modify a curve based on the physical/mathematical reality that the curve represents. When using the monotonic_derivative library, keep in mind that the goal is to create a more realistic or physically plausible representation of the original curve while preserving its essential characteristics. This can be particularly useful in applications where the original data may be subject to noise or other inaccuracies, and a more idealized or smoothed curve is required for analysis or presentation purposes.

### Contributing

We welcome contributions to the Monotonic Derivative library! Please feel free to submit pull requests for bug fixes, new features, or improvements to the code or documentation.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
