import scipy
import math

# TO DO
# Need to split def prep_ISSI_data from get_magnetogram or sth like that
# Need to decide on general format we want data to be given, probably as
# Bx = np.array(ny, nx) and By = np.array(ny, nx) at z = 0 (Photosphere)
# Additionally need pixelsize in km


def get_magnetogram(path):
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

    print(data["info_unit"])
    print(data["info_pixel"])
    # print(data['info_boundary'])
    print(data["info_array"])

    pixelsize = float(input("Pixelsize in km?"))

    bx_xlen = data_bx.shape[1]
    bx_ylen = data_bx.shape[0]
    by_xlen = data_by.shape[1]
    by_ylen = data_bx.shape[0]
    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
        print("x lengths of data do not match")
        raise ValueError
    if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
        print("y lengths of data do not match")
        raise ValueError
    else:
        nresol_x = bx_xlen  # Data resolution in x direction
        nresol_y = bx_ylen  # Data resolution in y direction

    L = 1.0

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


def get_magnetogram_SOAR(data):
    data_bz = data

    # pixelsize_km = float(input("Pixelsize for magnetogram in km?"))
    pixelsize_z_km = float(input("Pixelsize for vertical direction in km?"))

    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #     print("x lengths of data do not match")
    #     raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #     print("y lengths of data do not match")
    #     raise ValueError
    # else:
    #     nresol_x = bx_xlen  # Data resolution in x direction
    #     nresol_y = bx_ylen  # Data resolution in y direction

    nresol_x = bz_xlen
    nresol_y = bz_ylen
    L = 1.0

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
        10000.0 / pixelsize_z_km
    )  # Artifical upper boundary at 10Mm outside of corona
    z0_index = math.floor(2000.0 / pixelsize_z_km)  # Height of Transition Region at 2Mm

    if xmax == L:
        # zmax = nresol_z * pixelsize_z_km / (nresol_x * pixelsize_km)
        # z0 = z0_index * pixelsize_z_km / (nresol_x * pixelsize_km)
        zmax = nresol_z / nresol_x
        z0 = z0_index / nresol_x
    if ymax == L:
        # zmax = nresol_z * pixelsize_z_km / (nresol_y * pixelsize_km)
        # z0 = (z0_index * pixelsize_z_km) / (nresol_y * pixelsize_km)
        zmax = nresol_z / nresol_y
        z0 = z0_index / nresol_y

    pixelsize_z = abs(zmax - zmin) / nresol_z  # Data pixel size in z direction

    # pixelsize_z = pixelsize_x * pixelsize_z_km / pixelsize_km
    # if pixelsize_z != pixelsize_x:
    #    print("nresol_z and zmax do not match")
    #    raise ValueError

    nf_max = min(nresol_x, nresol_y) - 1

    return [
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
