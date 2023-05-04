from setuptools import setup
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="monotonic derivative",
    version="0.3",
    description="Monotonic Derivative is a Python library designed to modify real-life data to ensure that the specified degree derivative of the cubic spline is always monotonically increasing or decreasing. This library can be particularly useful in applications where the derivatives of the given data must follow specific monotonicity constraints, such as in scientific modeling or engineering applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adam Wecker",
    author_email="adam.wecker0@gmail.com",
    url="https://github.com/A-Wpro/monotonic_derivative",
    packages=["monotonic_derivative"],
    install_requires=["numpy", "scipy", "matplotlib", "imageio"],
    classifiers=[
        # See https://pypi.org/classifiers/ for a full list of classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
