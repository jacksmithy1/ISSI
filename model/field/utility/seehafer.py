import numpy as np


def mirror_magnetogram(
    data_bz,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    nresol_x: np.int16,
    nresol_y: np.int16,
):
    b_arr = np.zeros(
        (int(2 * nresol_y), int(2 * nresol_x))
    )  # [0:2*nresol_y,0:2*nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    if xmin != 0.0 or ymin != 0.0 or not (xmax > 0.0 or ymax > 0.0):
        print("Magneotgram not centred at origin and in correct quadrant")
        raise ValueError

    # Seehafer mirroring

    for ix in range(0, nresol_x):
        for iy in range(0, nresol_y):
            b_arr[nresol_y + iy, nresol_x + ix] = data_bz[iy, ix]
            b_arr[nresol_y + iy, ix] = -data_bz[iy, nresol_x - 1 - ix]
            b_arr[iy, nresol_x + ix] = -data_bz[nresol_y - 1 - iy, ix]
            b_arr[iy, ix] = data_bz[nresol_y - 1 - iy, nresol_x - 1 - ix]

    return b_arr
