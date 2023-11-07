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

# TO DO
# Extract boundary magnetic field vector from Magnetogram
# Gar nicht mal so einfach

# Magnetic field B Line Of Sight character
path_blos: str = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)

# plot_magnetogram.plot_magnetogram_boundary(image_data, 2048, 2048)
# plot_magnetogram.plot_magnetogram_boundary(photo_data, 800, 500)

data = read_fits_SOAR(path_blos)

# BFieldvec_Seehafer = np.load('field_data_potential.npy')

data_bz: np.ndarray[np.float64] = data[0]
nresol_x: np.int16 = data[1]
nresol_y: np.int16 = data[2]
nresol_z: np.int16 = data[3]
pixelsize_x: np.float64 = data[4]
pixelsize_y: np.float64 = data[5]
pixelsize_z: np.float64 = data[6]
nf_max: np.int16 = data[7]
xmin: np.float64 = data[8]
xmax: np.float64 = data[9]
ymin: np.float64 = data[10]
ymax: np.float64 = data[11]
zmin: np.float64 = data[12]
zmax: np.float64 = data[13]
z0: np.float64 = data[14]

# print(nresol_x, nresol_y, nresol_z)
# print(xmin, ymin, zmin)
# print(xmax, ymax, zmax)
# print(pixelsize_x, pixelsize_y, pixelsize_z)
# print(z0)
# print(nf_max)

a: np.float64 = 0.24
alpha: np.float64 = 0.5
b: np.float64 = 1.0

T_photosphere = 5600.0  # temperature photosphere in Kelvin
T_corona = 2.0 * 10.0**6.0  # temperature corona in Kelvin

h1 = 0.0001  # Initial step length for fieldline3D
eps = 1.0e-8  # Tolerance to which we require point on field line known for fieldline3D
hmin = 0.0  # Minimum step length for fieldline3D
hmax = 1.0  # Maximum step length for fieldline3D
deltaz = z0 / 10.0  # Width of transitional region ca. 200km

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
    nf_max,
    a,
    b,
    alpha,
)
