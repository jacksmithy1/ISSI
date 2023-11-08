# import astropy
# from astropy.io import fits
# import sunpy_soar
# from sunpy.net import Fido, attrs as a
# from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from plot.plot_magnetogram import (
    plot_fieldlines_grid,
    plot_magnetogram_boundary,
    plot_magnetogram_boundary_3D,
)
from model.field.bfield_model import magnetic_field
from load.read_file import read_fits_SOAR
from classes.clsmod import DataBz
import time

start_time = time.time()

# TO DO
# Extract boundary magnetic field vector from Magnetogram
# Gar nicht mal so einfach

# Magnetic field B Line Of Sight character
path_blos: str = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)

# plot_magnetogram.plot_magnetogram_boundary(image_data, 2048, 2048)
# plot_magnetogram.plot_magnetogram_boundary(photo_data, 800, 500)

data: DataBz = read_fits_SOAR(path_blos)

# BFieldvec_Seehafer = np.load('field_data_potential.npy')

data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_z
nresol_x: int = data.nresol_x
nresol_y: int = data.nresol_y
nresol_z: int = data.nresol_z
pixelsize_x: np.float64 = data.pixelsize_x
pixelsize_y: np.float64 = data.pixelsize_y
pixelsize_z: np.float64 = data.pixelsize_z
nf_max: int = data.nf_max
xmin: np.float64 = data.xmin
xmax: np.float64 = data.xmax
ymin: np.float64 = data.ymin
ymax: np.float64 = data.ymax
zmin: np.float64 = data.zmin
zmax: np.float64 = data.zmax
z0: np.float64 = data.z0

a: float = 0.24
alpha: float = 0.5
b: float = 1.0

h1: float = 0.0001  # Initial step length for fieldline3D
eps: float = 1.0e-8
# Tolerance to which we require point on field line known for fieldline3D
hmin: float = 0.0  # Minimum step length for fieldline3D
hmax: float = 1.0  # Maximum step length for fieldline3D

deltaz: np.float64 = np.float64(
    z0 / 10.0
)  # z0 at 2Mm so widht of transition region = 200km

B_Seehafer = magnetic_field(
    data_bz,
    z0,
    deltaz,
    a,
    b,
    alpha,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    nresol_x,
    nresol_y,
    nresol_z,
    pixelsize_x,
    pixelsize_y,
    nf_max,
)
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")

# b_back_test = np.zeros((2 * nresol_y, 2 * nresol_x))
# b_back_test = B_Seehafer[:, :, 0, 2]
# plot_magnetogram_boundary_3D(
#    b_back_test, nresol_x, nresol_y, -xmax, xmax, -ymax, ymax, zmin, zmax
# )

plot_fieldlines_grid(
    B_Seehafer,
    h1,
    hmin,
    hmax,
    eps,
    nresol_x,
    nresol_y,
    nresol_z,
    -xmax,
    xmax,
    -ymax,
    ymax,
    zmin,
    zmax,
)
