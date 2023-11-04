"""Setup script."""

import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="issi",
    version=__version__,
    author="Lilli Nadol",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "issi",
        "numba",
        "astropy",
        "sunpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
