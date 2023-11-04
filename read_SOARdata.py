import astropy
from astropy.io import fits
import sunpy_soar
from sunpy.net import Fido, attrs as a
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import plot_magnetogram
import get_data
import Seehafer
import BField_model


# TO DO
# Extract boundary magnetic field vector from Magnetogram
# Gar nicht mal so einfach

# Magnetic field B Line Of Sight character
path_blos = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)

with fits.open(path_blos) as data:
    # data.info()
    image_data = fits.getdata(path_blos, ext=0)
    # print(image_data.shape)
    photo_data = image_data[500:1000, 400:1200]
    # with open(
    #    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01_HEADER.txt",
    #    "w",
    # ) as f:
    #    for d in data:
    #        f.write(repr(d.header))

plot_magnetogram.plot_magnetogram_boundary(photo_data, 800, 500)

data = get_data.get_magnetogram_SOAR(photo_data)

# BFieldvec_Seehafer = np.load('field_data_potential.npy')

data_bz = data[0]
nresol_x = data[1]
nresol_y = data[2]
nresol_z = data[3]
pixelsize_x = data[4]
pixelsize_y = data[5]
pixelsize_z = data[6]
nf_max = data[7]
xmin = data[8]
xmax = data[9]
ymin = data[10]
ymax = data[11]
zmin = data[12]
zmax = data[13]
z0 = data[14]

print(nresol_x, nresol_y, nresol_z)
print(xmin, ymin, zmin)
print(xmax, ymax, zmax)
print(pixelsize_x, pixelsize_y, pixelsize_z)
print(z0)
print(nf_max)

a = 0.0
alpha = 0.0
b = 1.0

T_photosphere = 5600.0  # temperature photosphere in Kelvin
T_corona = 2.0 * 10.0**6.0  # temperature corona in Kelvin

h1 = 0.0001  # Initial step length for fieldline3D
eps = 1.0e-8  # Tolerance to which we require point on field line known for fieldline3D
hmin = 0.0  # Minimum step length for fieldline3D
hmax = 1.0  # Maximum step length for fieldline3D
deltaz = z0 / 10.0  # Width of transitional region ca. 200km

data_bz_Seehafer = Seehafer.mirror_magnetogram(
    data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
)

# plot_magnetogram.plot_magnetogram_boundary(data_bz_Seehafer, 2 * nresol_x, 2 * nresol_y)

B_Seehafer = BField_model.get_magnetic_field(
    data_bz_Seehafer,
    z0,
    deltaz,
    a,
    b,
    alpha,
    -xmax,
    xmax,
    -ymax,
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
# plot_magnetogram.plot_magnetogram_boundary_3D(
#    b_back_test, nresol_x, nresol_y, -xmax, xmax, -ymax, ymax, zmin, zmax
# )

plot_magnetogram.plot_fieldlines_grid(
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
    a,
    b,
    alpha,
    nf_max,
)
