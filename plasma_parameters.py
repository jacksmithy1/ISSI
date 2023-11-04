import BField_model
import numpy as np


def bpressure(z, z0, deltaz, h, T0, T1):
    q1 = deltaz / (2.0 * h * (1.0 + T1 / T0))
    q2 = deltaz / (2.0 * h * (1.0 - T1 / T0))
    q3 = deltaz * (T1 / T0) / (2.0 * h * (1.0 - (T1 / T0) ** 2))

    p1 = (
        2.0
        * np.exp(-2.0 * (z - z0) / deltaz)
        / (1.0 * np.exp(-2.0 * (z - z0) / deltaz))
        / (1.0 + np.tanh(z0 / deltaz))
    )
    p2 = (1.0 - np.tanh(z0 / deltaz)) / (1.0 + np.tanh((z - z0) / deltaz))
    p3 = (1.0 + T1 / T0 * np.tanh((z - z0) / deltaz)) / (
        1.0 - T1 / T0 * np.tanh(z0 / deltaz)
    )

    return p1**q1 * p2**q2 * p3**q3


def btemp(z, z0, deltaz, T0, T1):
    return T0 + T1 * np.tanh((z - z0) / deltaz)


def bdensity(z, z0, deltaz, T0, T1, h):
    temp0 = btemp(0.0, z0, deltaz, T0, T1)
    return bpressure(z, z0, deltaz, h, T1, T0) / btemp(z, z0, deltaz, T0, T1) * temp0


def pres(x, y, z, z0, deltaz, a, b, mu0, bz):
    Bzsqr = bz[y, x, z] ** 2.0
    return bpressure(z) + Bzsqr * BField_model.f(z, z0, deltaz, a, b) / (2.0 * mu0)


def den(x, y, z, z0, deltaz, a, b, bx, by, bz, dBz, beta0, h, T0, T_photosphere):
    Bx = bx[y, x, z]
    By = by[y, x, z]
    Bz = bz[y, x, z]
    dBzdx = dBz[y, x, z, 0]
    dBzdy = dBz[y, x, z, 1]
    dBzdz = dBz[y, x, z, 1]
    BdotgradBz = Bx * dBzdx + By * dBzdy + Bz * dBzdz
    return (
        0.5 * beta0 / h * T0 / T_photosphere * bdensity(z)
        + BField_model.dfdz(z, z0, deltaz, a, b) * Bz**2.0 / 2.0
        + BField_model.f(z, z0, deltaz, a, b) * BdotgradBz
    )


# Something with derivative of Bz ugh
# B dot gradBz
# need gradBz
