import issi.BField_model as BField_model
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
    temp0 = btemp(z, z0, deltaz, T0, T1)
    return bpressure(z, z0, deltaz, h, T1, T0) / btemp(z, z0, deltaz, T0, T1) * temp0


def pres(ix, iy, iz, z, z0, deltaz, a, b, beta0, bz, h, T0, T1):
    Bzsqr = bz[iy, ix, iz] ** 2.0
    return (
        0.5 * beta0 * bpressure(z, z0, deltaz, h, T0, T1)
        + Bzsqr * BField_model.f(z, z0, deltaz, a, b) / 2.0
    )


def den(
    ix, iy, iz, z, z0, deltaz, a, b, bx, by, bz, dBz, beta0, h, T0, T1, T_photosphere
):
    Bx = bx[iy, ix, iz]
    By = by[iy, ix, iz]
    Bz = bz[iy, ix, iz]
    dBzdx = dBz[iy, ix, iz, 0]
    dBzdy = dBz[iy, ix, iz, 1]
    dBzdz = dBz[iy, ix, iz, 1]
    BdotgradBz = Bx * dBzdx + By * dBzdy + Bz * dBzdz
    return (
        0.5 * beta0 / h * T0 / T_photosphere * bdensity(z, z0, deltaz, T0, T1, h)
        + BField_model.dfdz(z, z0, deltaz, a, b) * Bz**2.0 / 2.0
        + BField_model.f(z, z0, deltaz, a, b) * BdotgradBz
    )


def temp(
    ix, iy, iz, z, z0, deltaz, a, b, bx, by, bz, dBz, beta0, h, T0, T1, T_photosphere
):
    p = pres(ix, iy, iz, z, z0, deltaz, a, b, beta0, bz, h, T0, T1)
    d = den(
        ix,
        iy,
        iz,
        z,
        z0,
        deltaz,
        a,
        b,
        bx,
        by,
        bz,
        dBz,
        beta0,
        h,
        T0,
        T1,
        T_photosphere,
    )
    return p / d


# Something with derivative of Bz ugh
# B dot gradBz
# need gradBz
