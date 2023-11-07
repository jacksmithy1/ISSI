import numpy as np
from model.field.utility.fft import fft_coeff_seehafer
from model.field.utility.seehafer import mirror_magnetogram
from model.field.utility.poloidal import phi, phi_low, dphidz, dphidz_low


def magnetic_field(
    data_bz,
    z0: np.float64,
    deltaz: np.float64,
    a: np.float64,
    b: np.float64,
    alpha: np.float64,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: np.int16,
    nresol_y: np.int16,
    nresol_z: np.int16,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: np.int16,
    ffunc: str = "Neukirch",
):
    length_scale = 2.0  # Normalising length scale for Seehafer
    length_scale_x = 2.0 * nresol_x * pixelsize_x
    # Length scale in x direction for Seehafer
    length_scale_y = 2.0 * nresol_y * pixelsize_y
    # Length scale in y direction for Seehafer
    length_scale_x_norm = length_scale_x / length_scale
    # Normalised length scale in x direction for Seehafer
    length_scale_y_norm = length_scale_y / length_scale
    # Normalised length scale in y direction for Seehafer

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        print("Magnotgram not centred at origin")
        raise ValueError
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        print("Magnetrogram in wrong quadrant of Seehafer mirroring")
        raise ValueError

    x_arr = np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    y_arr = np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    ratiodzls = deltaz / length_scale  # Normalised deltaz

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf_max) * np.pi / length_scale_x_norm  # [0:nf_max]
    ky_arr = np.arange(nf_max) * np.pi / length_scale_y_norm  # [0:nf_max]

    one_arr = 0.0 * np.arange(nf_max) + 1.0

    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale) ** 2

    p_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    q_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)

    data_bz_seehafer = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm, signal = fft_coeff_seehafer(
        data_bz_seehafer, k2_arr, 2 * nresol_x, 2 * nresol_y, nf_max
    )

    phi_arr = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]
    dphidz_arr = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]

    for iy in range(0, nf_max):
        for ix in range(0, nf_max):
            q = q_arr[iy, ix]
            p = p_arr[iy, ix]
            for iz in range(0, nresol_z):
                z = z_arr[iz]
                if ffunc == "Low":
                    phi_arr[iy, ix, iz] = phi_low(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz_low(z, p, q, z0, deltaz)
                if ffunc == "Neukirch":
                    phi_arr[iy, ix, iz] = phi(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz(z, p, q, z0, deltaz)

    b_arr = np.zeros((2 * nresol_y, 2 * nresol_x, nresol_z, 3))

    sin_x = np.zeros((2 * nresol_x, nf_max))  # [0:2*nresol_x, 0:nf_max]
    sin_y = np.zeros((2 * nresol_y, nf_max))  # [0:2*nresol_y, 0:nf_max]
    cos_x = np.zeros((2 * nresol_x, nf_max))  # [0:2*nresol_x, 0:nf_max]
    cos_y = np.zeros((2 * nresol_y, nf_max))  # [0:2*nresol_y, 0:nf_max]

    sin_x = np.sin(np.outer(kx_arr, x_arr))
    sin_y = np.sin(np.outer(ky_arr, y_arr))
    cos_x = np.cos(np.outer(kx_arr, x_arr))
    cos_y = np.cos(np.outer(ky_arr, y_arr))

    for iz in range(0, nresol_z):
        coeffs = np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

        coeffs1 = np.multiply(np.multiply(anm, dphidz_arr[:, :, iz]), ky_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        coeffs2 = alpha * np.multiply(np.multiply(anm, phi_arr[:, :, iz]), kx_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs1, sin_x)) - np.matmul(
            sin_y.T, np.matmul(coeffs2, cos_x)
        )
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

        # Fieldline3D program was written for order B = [Bx,By,Bz] with indexing [ix,iy,iz] but here we have indexing [iy,ix,iz]
        # so in order to be consistent we have to switch to order B = [Bx,By,Bz] such that fieldline3D program treats X and Y as Y and X consistently

        coeffs3 = np.multiply(np.multiply(anm, dphidz_arr[:, :, iz]), kx_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        coeffs4 = alpha * np.multiply(np.multiply(anm, phi_arr[:, :, iz]), ky_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs3, cos_x)) + np.matmul(
            cos_y.T, np.matmul(coeffs4, sin_x)
        )
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

    return b_arr


def bz_partial_derivatives(
    data_bz,
    z0: np.float64,
    deltaz: np.float64,
    a: np.float64,
    b: np.float64,
    alpha: np.float64,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: np.int16,
    nresol_y: np.int16,
    nresol_z: np.int16,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: np.int16,
    ffunc: str = "Neukirch",
):
    length_scale = 2.0  # Normalising length scale for Seehafer
    length_scale_x = 2.0 * nresol_x * pixelsize_x
    # Length scale in x direction for Seehafer
    length_scale_y = 2.0 * nresol_y * pixelsize_y
    # Length scale in y direction for Seehafer
    length_scale_x_norm = length_scale_x / length_scale
    # Normalised length scale in x direction for Seehafer
    length_scale_y_norm = length_scale_y / length_scale
    # Normalised length scale in y direction for Seehafer

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        print("Magnotgram not centred at origin")
        raise ValueError
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        print("Magnetrogram in wrong quadrant of Seehafer mirroring")
        raise ValueError

    x_arr = np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    y_arr = np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    ratiodzls = deltaz / length_scale  # Normalised deltaz

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf_max) * np.pi / length_scale_x_norm  # [0:nf_max]
    ky_arr = np.arange(nf_max) * np.pi / length_scale_y_norm  # [0:nf_max]

    one_arr = 0.0 * np.arange(nf_max) + 1.0

    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale) ** 2

    p_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    q_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)

    data_bz_seehafer = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm, signal = fft_coeff_seehafer(
        data_bz_seehafer, k2_arr, 2 * nresol_x, 2 * nresol_y, nf_max
    )

    phi_arr = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]
    dphidz_arr = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]

    for iy in range(0, nf_max):
        for ix in range(0, nf_max):
            q = q_arr[iy, ix]
            p = p_arr[iy, ix]
            for iz in range(0, nresol_z):
                z = z_arr[iz]
                if ffunc == "Low":
                    phi_arr[iy, ix, iz] = phi_low(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz_low(z, p, q, z0, deltaz)
                if ffunc == "Neukirch":
                    phi_arr[iy, ix, iz] = phi(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz(z, p, q, z0, deltaz)

    bz_derivs = np.zeros((2 * nresol_y, 2 * nresol_x, nresol_z, 3))

    sin_x = np.zeros((2 * nresol_x, nf_max))  # [0:2*nresol_x, 0:nf_max]
    sin_y = np.zeros((2 * nresol_y, nf_max))  # [0:2*nresol_y, 0:nf_max]
    cos_x = np.zeros((2 * nresol_x, nf_max))  # [0:2*nresol_x, 0:nf_max]
    cos_y = np.zeros((2 * nresol_y, nf_max))  # [0:2*nresol_y, 0:nf_max]

    sin_x = np.sin(np.outer(kx_arr, x_arr))
    sin_y = np.sin(np.outer(ky_arr, y_arr))
    cos_x = np.cos(np.outer(kx_arr, x_arr))
    cos_y = np.cos(np.outer(ky_arr, y_arr))

    for iz in range(0, nresol_z):
        coeffs = np.multiply(np.multiply(k2_arr, dphidz_arr[:, :, iz]), anm)
        bz_derivs[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))

        coeffs2 = np.multiply(
            np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm), kx_grid
        )
        bz_derivs[:, :, iz, 0] = np.matmul(sin_y.T, np.matmul(coeffs2, cos_x))

        coeffs3 = np.multiply(
            np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm),
            ky_grid,
        )
        bz_derivs[:, :, iz, 1] = np.matmul(cos_y.T, np.matmul(coeffs3, sin_x))

    return bz_derivs
