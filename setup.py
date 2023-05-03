from setuptools import setup

setup(
    name='monotonic derivative',
    version='0.1',
    description='Monotonic Derivative is a Python library designed to modify real-life data to ensure that the specified degree derivative of the cubic spline is always monotonically increasing or decreasing. This library can be particularly useful in applications where the derivatives of the given data must follow specific monotonicity constraints, such as in scientific modeling or engineering applications.',
    author='Adam Wecker',
    packages=['monotonic_derivative'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'imageio'],
)