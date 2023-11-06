import numpy as np
from datetime import datetime
import time
import scipy
import fft
import seehafer


def f(z, a, b=None, kappa=None, z0=None, deltaz=None, height_profile="tanh"):
    if height_profile == "tanh":
        return a * (1.0 - b * np.tanh((z - z0) / deltaz))
    if height_profile == "exp":
        return a * np.exp(-kappa * z)


def dfdz(z, a, b=None, kappa=None, z0=None, deltaz=None, height_profile="tanh"):
    if height_profile == "tanh":
        return -a * b / (deltaz * np.cosh((z - z0) / deltaz))
    if height_profile == "exp":
        return -kappa * a * np.exp(-kappa * z)


def phi(z, p, q, z0, deltaz, height_profile="tanh"):
    if height_profile == "tanh":
        rplus = p / deltaz
        rminus = q / deltaz
        R = rminus / rplus
        D = np.cosh(2.0 * rplus * z0) + R * np.sinh(2.0 * rplus * z0)
        if z - z0 < 0.0:
            return (
                np.cosh(2.0 * rplus * (z0 - z)) + R * np.sinh(2.0 * rplus * (z0 - z))
            ) / D
        else:
            return np.exp(-2.0 * rminus * (z - z0)) / D
    if height_profile == "exp":
        return (
            scipy.special.jv(p, q * np.exp(-z / (2.0 * deltaz)))
        ) / scipy.special.jv(p, q)


def dphidz(z, p, q, z0, deltaz, height_profile="tanh"):
    if height_profile == "tanh":
        rplus = p / deltaz
        rminus = q / deltaz
        R = rminus / rplus
        D = np.cosh(2.0 * rplus * z0) + R * np.sinh(2.0 * rplus * z0)

        if z - z0 < 0.0:
            return (
                -2.0
                * rplus
                * (
                    np.sinh(2.0 * rplus * (z0 - z))
                    + R * np.cosh(2.0 * rplus * (z0 - z))
                )
            ) / D
        else:
            return -2.0 * rminus * np.exp(-2.0 * rminus * (z - z0)) / D
    if height_profile == "exp":
        return (
            (
                q
                * np.exp(-z / (2.0 * deltaz))
                * scipy.special.jv(p + 1.0, q * np.exp(-z / (2.0 * deltaz)))
                - p * scipy.special.jv(p, q * np.exp(-z / (2.0 * deltaz)))
            )
            / (2.0 * deltaz)
        ) / scipy.special.jv(p, q)


def bfield(
    data_bz,
    z0,
    deltaz,
    a,
    b,
    alpha,
    boxedges,
    resolutions,
    pixelsizes,
    nf_max,
    L,
    height_profile="tanh",
):
    xmin = boxedges[0, 0]
    xmax = boxedges[0, 1]
    ymin = boxedges[1, 0]
    ymax = boxedges[1, 1]
    zmin = boxedges[2, 0]
    zmax = boxedges[2, 1]
    nresol_x = int(resolutions[0])
    nresol_y = int(resolutions[1])
    nresol_z = int(resolutions[2])
    pixelsize_x = pixelsizes[0]
    pixelsize_y = pixelsizes[1]

    L_Seehafer = 2.0 * L  # Normalising length scale for Seehafer
    L_x_Seehafer = (
        2.0 * nresol_x * pixelsize_x
    )  # Length scale in x direction for Seehafer
    L_y_Seehafer = (
        2.0 * nresol_y * pixelsize_y
    )  # Length scale in y direction for Seehafer
    L_x_Seehafer_norm = (
        L_x_Seehafer / L_Seehafer
    )  # Normalised length scale in x direction for Seehafer
    L_y_Seehafer_norm = (
        L_y_Seehafer / L_Seehafer
    )  # Normalised length scale in y direction for Seehafer

    # X and Y arrays for Seehafer (double length, different start point), Z stays the same

    X_Seehafer = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    Y_Seehafer = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    Z = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    ratiodzL_Seehafer = deltaz / L_Seehafer  # Normalised deltaz

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr_Seehafer = np.arange(nf_max) * np.pi / L_x_Seehafer_norm  # [0:nf_max]
    ky_arr_Seehafer = np.arange(nf_max) * np.pi / L_y_Seehafer_norm  # [0:nf_max]

    one_arr_Seehafer = 0.0 * np.arange(nf_max) + 1.0

    ky_arr_Seehafer_mat = np.outer(
        ky_arr_Seehafer, one_arr_Seehafer
    )  # [0:nf_max, 0:nf_max]
    kx_arr_Seehafer_mat = np.outer(
        one_arr_Seehafer, kx_arr_Seehafer
    )  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr_Seehafer = np.outer(ky_arr_Seehafer**2, one_arr_Seehafer) + np.outer(
        one_arr_Seehafer, kx_arr_Seehafer**2
    )  # [0:nf_max, 0:nf_max]
    k2_arr_Seehafer[0, 0] = (np.pi / L_Seehafer) ** 2

    p_arr_Seehafer = (
        0.5
        * ratiodzL_Seehafer
        * np.sqrt(k2_arr_Seehafer * (1.0 - a - a * b) - alpha**2)
    )  # [0:nf_max, 0:nf_max]
    q_arr_Seehafer = (
        0.5
        * ratiodzL_Seehafer
        * np.sqrt(k2_arr_Seehafer * (1.0 - a + a * b) - alpha**2)
    )

    data_bz_Seehafer = seehafer.mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )

    anm, signal = fft.fft_coeff_Seehafer(
        data_bz_Seehafer, k2_arr_Seehafer, 2 * nresol_x, 2 * nresol_y, nf_max
    )

    Phifunc = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]
    DPhifuncDz = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]

    for iy in range(0, nf_max):
        for ix in range(0, nf_max):
            q = q_arr_Seehafer[iy, ix]
            p = p_arr_Seehafer[iy, ix]
            for iz in range(0, nresol_z):
                z = Z[iz]
                Phifunc[iy, ix, iz] = phi(
                    z, p, q, z0, deltaz, height_profile=height_profile
                )
                DPhifuncDz[iy, ix, iz] = dphidz(
                    z, p, q, z0, deltaz, height_profile=height_profile
                )

    BFieldvec_Seehafer = np.zeros(
        (2 * nresol_y, 2 * nresol_x, nresol_z, 3)
    )  # [0:2*nresol_y,0:2*nresol_x, 0:nresol_z]

    sin_x = np.zeros((2 * nresol_x, nf_max))  # [0:2*nresol_x, 0:nf_max]
    sin_y = np.zeros((2 * nresol_y, nf_max))  # [0:2*nresol_y, 0:nf_max]
    cos_x = np.zeros((2 * nresol_x, nf_max))  # [0:2*nresol_x, 0:nf_max]
    cos_y = np.zeros((2 * nresol_y, nf_max))  # [0:2*nresol_y, 0:nf_max]

    sin_x = np.sin(np.outer(kx_arr_Seehafer, X_Seehafer))
    sin_y = np.sin(np.outer(ky_arr_Seehafer, Y_Seehafer))
    cos_x = np.cos(np.outer(kx_arr_Seehafer, X_Seehafer))
    cos_y = np.cos(np.outer(ky_arr_Seehafer, Y_Seehafer))

    for iz in range(0, nresol_z):
        Coeff_matrix = np.multiply(
            np.multiply(k2_arr_Seehafer, Phifunc[:, :, iz]), anm
        )  # Componentwise multiplication, [0:nf_max, 0:nf_max]
        dummymat1 = np.matmul(
            Coeff_matrix, sin_x
        )  # [0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:nf_max, 0:2*nresol_x]
        dummymat2 = np.matmul(
            sin_y.T, dummymat1
        )  # [0:2*nresol_y, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:2*nresol_y, 0:2*nresol_x]
        BFieldvec_Seehafer[:, :, iz, 2] = dummymat2  # [0:2*nresol_y, 0:2*nresol_x]

        Coeff_matrix2 = np.multiply(
            np.multiply(anm, DPhifuncDz[:, :, iz]), ky_arr_Seehafer_mat
        )  # Componentwise multiplication, [0:nf_max, 0:nf_max]
        Coeff_matrix3 = alpha * np.multiply(
            np.multiply(anm, Phifunc[:, :, iz]), kx_arr_Seehafer_mat
        )  # Componentwise multiplication, [0:nf_max, 0:nf_max]
        dummymat3 = np.matmul(
            Coeff_matrix2, sin_x
        )  # [0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:nf_max, 0:2*nresol_x]
        dummymat4 = np.matmul(
            Coeff_matrix3, cos_x
        )  # [0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:nf_max, 0:2*nresol_x]
        dummymat5 = np.matmul(
            cos_y.T, dummymat3
        )  # [0:2*nresol_y, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:2*nresol_y, 0:2*nresol_x]
        dummymat6 = np.matmul(
            sin_y.T, dummymat4
        )  # [0:2*nresol_y, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:2*nresol_y, 0:2*nresol_x]
        BFieldvec_Seehafer[:, :, iz, 0] = dummymat5 - dummymat6

        # Fieldline3D program was written for order B = [Bx,By,Bz] with indexing [ix,iy,iz] but here we have indexing [iy,ix,iz]
        # so in order to be consistent we have to switch to order B = [Bx,By,Bz] such that fieldline3D program treats X and Y as Y and X consistently

        Coeff_matrix4 = np.multiply(
            np.multiply(anm, DPhifuncDz[:, :, iz]), kx_arr_Seehafer_mat
        )  # Componentwise multiplication, [0:nf_max, 0:nf_max]
        Coeff_matrix5 = alpha * np.multiply(
            np.multiply(anm, Phifunc[:, :, iz]), ky_arr_Seehafer_mat
        )  # Componentwise multiplication, [0:nf_max, 0:nf_max]
        dummymat7 = np.matmul(
            Coeff_matrix4, cos_x
        )  # [0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:nf_max, 0:2*nresol_x]
        dummymat8 = np.matmul(
            Coeff_matrix5, sin_x
        )  # [0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:nf_max, 0:2*nresol_x]
        dummymat9 = np.matmul(
            sin_y.T, dummymat7
        )  # [0:2*nresol_y, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:2*nresol_y, 0:2*nresol_x]
        dummymat10 = np.matmul(
            cos_y.T, dummymat8
        )  # [0:2*nresol_y, 0:nf_max]*[0:nf_max, 0:2*nresol_x] = [0:2*nresol_y, 0:2*nresol_x]
        BFieldvec_Seehafer[:, :, iz, 1] = dummymat9 + dummymat10

    return BFieldvec_Seehafer
