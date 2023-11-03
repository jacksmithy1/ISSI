import numpy as np


def mirror_magnetogram(data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y):
    data_bz_Seehafer = np.zeros(
        (2 * nresol_y, 2 * nresol_x)
    )  # [0:2*nresol_y,0:2*nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    if xmin != 0.0 or ymin != 0.0:
        print("xmin or ymin unequal 0")
        raise ValueError

    # Seehafer mirroring

    for ix in range(0, nresol_x):
        for iy in range(0, nresol_y):
            data_bz_Seehafer[nresol_y + iy, nresol_x + ix] = data_bz[iy, ix]
            data_bz_Seehafer[nresol_y + iy, ix] = -data_bz[iy, nresol_x - 1 - ix]
            data_bz_Seehafer[iy, nresol_x + ix] = -data_bz[nresol_y - 1 - iy, ix]
            data_bz_Seehafer[iy, ix] = data_bz[nresol_y - 1 - iy, nresol_x - 1 - ix]

    return data_bz_Seehafer
