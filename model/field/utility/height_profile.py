import numpy as np


def f(
    z: np.float64, z0: np.float64, deltaz: np.float64, a: float, b: float
) -> np.float64:
    return a * (1.0 - b * np.tanh((z - z0) / deltaz))


def f_low(z: np.float64, a: float, kappa: float) -> np.float64:
    return a * np.exp(-kappa * z)


def dfdz(
    z: np.float64, z0: np.float64, deltaz: np.float64, a: float, b: float
) -> np.float64:
    return -a * b / (deltaz * np.cosh((z - z0) / deltaz))


def dfdz_low(z: np.float64, a: float, kappa: float) -> np.float64:
    return -kappa * a * np.exp(-kappa * z)
