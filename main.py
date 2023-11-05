import get_data
import matplotlib.pyplot as plt
import numpy as np
import Seehafer
import plot_magnetogram
import BField_model
import datetime
import os

# TO DO

# Plasma Parameters
# optimise choice of a and alpha, switch case for potential, LFF and MHS
# Add Bessel or Neukirch parameter to calling function, also to file names

"""
B0      : amplitude of magnetic field strength
alpha   : linear force free parameter (in units of 1/L)
a       : amplitude parameter of f(z)
b       : parameter determining asymptotic value for z --> infinity
z0      : z-value of "transition" in current density
deltaz  : width of the transition in current density
T0      : temperature at z=z0
T1      : determines coronal temperature (T0+T1)
H       : pressure scale height (measured in units of basic length) based on T0
p0      : plasma pressure at z=0
rho0    : plasma mass density at z=0
nf_max  : number of Fourier modes in x and y
kx      : wave numbers in the x-direction (2*pi*n/L_x)
ky      : wave numbers in the y-direction (2*pi*m/L_y)
k2      : array of kx^2 + ky^2 values
p       : determined by k, a, b and alpha - sqrt(k^2*(1-a-a*b) - alpha^2)/2
q       : determined by k, a, b and alpha - sqrt(k^2*(1-a+a*b) - alpha^2)/2
anm     : array of Fourier coefficients of the sin(kx*x)*sin(ky*y) terms
bnm     : array of Fourier coefficients of the sin(kx*x)*cos(ky*y) terms
cnm     : array of Fourier coefficients of the cos(kx*x)*sin(ky*y) terms
dnm     : array of Fourier coefficients of the cos(kx*x)*cos(ky*y) terms
"""
a = 0.0
alpha = 0.0
b = 1.0

T_photosphere = 5600.0  # temperature photosphere in Kelvin
T_corona = 2.0 * 10.0**6.0  # temperature corona in Kelvin

h1 = 0.0001  # Initial step length for fieldline3D
eps = 1.0e-8  # Tolerance to which we require point on field line known for fieldline3D
hmin = 0.0  # Minimum step length for fieldline3D
hmax = 1.0  # Maximum step length for fieldline3D

# data = get_data.get_magnetogram('Analytic_boundary_data.sav')

data = get_data.get_magnetogram("RMHD_boundary_data.sav")

# BFieldvec_Seehafer = np.load('field_data_potential.npy')

data_bx = data[0]
data_by = data[1]
data_bz = data[2]
nresol_x = data[3]
nresol_y = data[4]
nresol_z = data[5]
pixelsize_x = data[6]
pixelsize_y = data[7]
pixelsize_z = data[8]
nf_max = data[9]
xmin = data[10]
xmax = data[11]
ymin = data[12]
ymax = data[13]
zmin = data[14]
zmax = data[15]
z0 = data[16]

deltaz = z0 / 10.0  # Width of transitional region ca. 200km

T0 = (T_photosphere + T_corona * np.tanh(z0 / deltaz)) / (1.0 + np.tanh(z0 / deltaz))
T1 = (T_corona - T_photosphere) / (1.0 + np.tanh(z0 / deltaz))
g_solar = 274.0  # in m/s^2
mbar = 1.0  # photospheric mean molecular weight, as multiples of proton mass
H = 1.3807 * T0 / (mbar * 1.6726 * g_solar) * 0.001
rho0 = 3.0 - 4  # photospheric mass density in kg/m^3
B0 = 500.0  # normalising photospheric B-field in Gauss
p0 = (
    1.3807 * T_photosphere * rho0 / (mbar * 1.6726) * 1.0
)  # photospheric plasma pressure in Pascal
pB0 = 3.9789 - 3 * B0**2.0  # photospheric magnetic pressure in Pascal
beta0 = p0 / pB0  # photospheric plasma beta

# plot_magnetogram.plot_magnetogram_boundary(data_bz, nresol_x, nresol_y)

data_bz_Seehafer = Seehafer.mirror_magnetogram(
    data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
)

# plot_magnetogram.plot_magnetogram_boundary(data_bz_Seehafer, 2*nresol_x, 2*nresol_y)

# plot_magnetogram.plot_magnetogram_boundary_3D(data_bz_Seehafer, nresol_x, nresol_y, -xmax, xmax, -ymax, ymax, zmin, zmax)


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

current_time = datetime.datetime.now()

dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

path = (
    "/Users/lilli/Desktop/ISSI_data/B_ISSI_RMHD_tanh_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(nf_max)
    + "_"
    + dt_string
    + ".npy"
)

with open(path, "wb") as file:
    np.save(file, B_Seehafer)

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
