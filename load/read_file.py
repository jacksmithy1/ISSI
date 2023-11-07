import scipy
import numpy as np
import math
from astropy.io import fits
from plot.plot_magnetogram import plot_magnetogram_boundary

# TO DO
# Need to split def prep_ISSI_data from get_magnetogram or sth like that
# Need to decide on general format we want data to be given, probably as
# Bx = np.array(ny, nx) and By = np.array(ny, nx) at z = 0 (Photosphere)
# Additionally need pixelsize in km


def get_magnetogram(path: str):
    data = scipy.io.readsav(path, python_dict=True, verbose=True)
    # data_bz = data['b2dz5'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # data_bx = data['b2dx5'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # data_by = data['b2dy5'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # scale_height = data['h3d']

    data_bz = data["b2dz"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    # print(data["info_unit"])
    # print(data["info_pixel"])
    # print(data['info_boundary'])
    # print(data["info_array"])

    pixelsize: np.float64 = float(input("Pixelsize in km?"))

    # bx_xlen: np.int16 = data_bx.shape[1]
    # bx_ylen: np.int16 = data_bx.shape[0]
    # by_xlen: np.int16 = data_by.shape[1]
    # by_ylen: np.int16 = data_bx.shape[0]
    bz_xlen: np.int16 = data_bz.shape[1]
    bz_ylen: np.int16 = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x: np.int16 = bz_xlen
    nresol_y: np.int16 = bz_ylen
    L: np.array64 = 1.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm

    if nresol_x < nresol_y:
        xmax = L  # Maximum value of x in data length scale, not in Mm
        ymax = nresol_y / nresol_x  # Maximum value of y in data length scale, not in Mm
    if nresol_y < nresol_x:
        ymax = L
        xmax = nresol_x / nresol_y
    if nresol_y == nresol_x:
        xmax = L
        ymax = L

    pixelsize_x = abs(xmax - xmin) / nresol_x  # Data pixel size in x direction
    pixelsize_y = abs(ymax - ymin) / nresol_y  # Data pixel size in y direction

    if pixelsize_x != pixelsize_y:
        print("directional pixel sizes of data do not match")
        raise ValueError

    nresol_z = math.floor(
        10000.0 / pixelsize
    )  # Artifical upper boundary at 10Mm outside of corona
    z0_index = math.floor(2000.0 / pixelsize)  # Height of Transition Region at 2Mm

    if xmax == L:
        zmax = nresol_z / nresol_x
        z0 = z0_index / nresol_x
    if ymax == L:
        zmax = nresol_z / nresol_y
        z0 = z0_index / nresol_y

    pixelsize_z = abs(zmax - zmin) / nresol_z  # Data pixel size in z direction

    if pixelsize_z != pixelsize_x:
        print("nresol_z and zmax do not match")
        raise ValueError

    nf_max = min(nresol_x, nresol_y) - 1

    return [
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    ]


def read_fits_SOAR(path: str, header=False):
    with fits.open(path) as data:
        # data.info()
        image = fits.getdata(path, ext=0)
        x_len = image.shape[0]
        y_len = image.shape[1]
        # plot_magnetogram_boundary(image, x_len, y_len)
        # x_start = int(input("First pixel x axis: "))
        # x_last = int(input("Last pixel x axis: "))
        # y_start = int(input("First pixel y axis: "))
        # y_last = int(input("Last pixel y axis: "))
        x_start = 400
        x_last = 1200
        y_start = 500
        y_last = 1000
        cut_image = image[y_start:y_last, x_start:x_last]
        # plot_magnetogram_boundary(cut_image, x_last - x_start, y_last - y_start)
        if header == True:
            with open(
                "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01_HEADER.txt",
                "w",
            ) as f:
                for d in data:
                    f.write(repr(d.header))
            print("File header has been printed to Desktop/SOAR/obs")
        hdr = data[0].header  # the primary HDU header
        dist = hdr["DSUN_OBS"]
        pixelsize_x_unit = hdr["CUNIT1"]
        pixelsize_y_unit = hdr["CUNIT2"]
        pixelsize_x_arcsec = hdr["CDELT1"]
        pixelsize_y_arcsec = hdr["CDELT2"]

        if not pixelsize_x_unit == pixelsize_y_unit:
            print("Pixelsize units not matchy-matchy")
            raise ValueError
        if not pixelsize_x_arcsec == pixelsize_y_arcsec:
            print("Data pixelsizes in x and y direction not matchy-matchy")
            raise ValueError
        else:
            pixelsize_radians = pixelsize_x_arcsec / 206265.0

    dist_km = dist / 1000.0
    pixelsize_km = math.floor(pixelsize_radians * dist_km)

    nresol_x = cut_image.shape[1]
    nresol_y = cut_image.shape[0]
    length_scale = 1.0  # L

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm

    if nresol_x < nresol_y:
        xmax = length_scale  # Maximum value of x in data length scale, not in Mm
        ymax = nresol_y / nresol_x  # Maximum value of y in data length scale, not in Mm
    if nresol_y < nresol_x:
        ymax = length_scale
        xmax = nresol_x / nresol_y
    if nresol_y == nresol_x:
        xmax = length_scale
        ymax = length_scale

    pixelsize_x = (
        abs(xmax - xmin) / nresol_x
    )  # Data pixel size in x direction in relation to xmin and xmax
    pixelsize_y = (
        abs(ymax - ymin) / nresol_y
    )  # Data pixel size in y direction in relation to ymin and ymax

    if pixelsize_x != pixelsize_y:
        print("Directional pixel sizes of data do not match")
        raise ValueError

    nresol_z = math.floor(
        10000.0 / (pixelsize_km / 4.0)
    )  # Artifical upper boundary at 10Mm from photosphere
    z0_index = math.floor(
        2000.0 / (pixelsize_km / 4.0)
    )  # Centre of region over which transition from NFF to FF takes place at 2Mm from photosphere

    if xmax == length_scale:
        zmax = nresol_z / nresol_x
        z0 = z0_index / nresol_x
    if ymax == length_scale:
        zmax = nresol_z / nresol_y
        z0 = z0_index / nresol_y

    pixelsize_z = (
        abs(zmax - zmin) / nresol_z
    )  # Data pixel size in z direction in relation to zmin and zmax

    # Calulate parameters in Mm for checking purposes

    # xmax_Mm = nresol_x * pixelsize_km / 1000.0
    # ymax_Mm = nresol_y * pixelsize_km / 1000.0
    # zmax_Mm = nresol_z * pixelsize_km / 1000.0

    # print("Mm", xmax_Mm, ymax_Mm, zmax_Mm)

    # if xmax == length_scale:
    # ratio_Mm_xy = ymax_Mm / xmax_Mm
    # ratio_Mm_xz = zmax_Mm / xmax_Mm
    # if ymax == length_scale:
    # ratio_Mm_xy = xmax_Mm / ymax_Mm
    # ratio_Mm_xz = zmax_Mm / ymax_Mm

    # print("ratio Mm", ratio_Mm_xy, ratio_Mm_xz)
    # pixelsize_z = pixelsize_x * pixelsize_z_km / pixelsize_km
    # if pixelsize_z != pixelsize_x:
    #    print("nresol_z and zmax do not match")
    #    raise ValueError

    nf_max = min(nresol_x, nresol_y)

    return [
        cut_image,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    ]
