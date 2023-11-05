import matplotlib.pyplot as plt
import numpy as np

from issi.Seehafer import mirror_magnetogram
from issi.BField_model import Bz_parderiv, get_magnetic_field
from issi.plasma_parameters import pres, den, temp
from issi.get_data import get_magnetogram

a = 0.0
alpha = 0.0
b = 1.0

T_photosphere = 5600.0  # temperature photosphere in Kelvin
T_corona = 2.0 * 10.0**6.0  # temperature corona in Kelvin

h1 = 0.0001  # Initial step length for fieldline3D
eps = 1.0e-8  # Tolerance to which we require point on field line known for fieldline3D
hmin = 0.0  # Minimum step length for fieldline3D
hmax = 1.0  # Maximum step length for fieldline3D

data = get_magnetogram("RMHD_boundary_data.sav")

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

data_bz_Seehafer = mirror_magnetogram(
    data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
)

dBzd = Bz_parderiv(
    data_bz_Seehafer,
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
    ffunc="Neukirch",
)

dBzdx = dBzd[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 1]
dBzdy = dBzd[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 0]
dBzdz = dBzd[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 2]

B_Seehafer = get_magnetic_field(
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
b_back = np.zeros((nresol_y, nresol_x))
b_back = B_Seehafer[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, 0, 2]
maxcoord = np.unravel_index(np.argmax(b_back, axis=None), b_back.shape)

iy = maxcoord[0]
ix = maxcoord[1]

Bx = B_Seehafer[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 1]
By = B_Seehafer[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 0]
Bz = B_Seehafer[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 2]

X = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
Y = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
Z = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

x_arr = X[nresol_x : 2 * nresol_x]
y_arr = Y[nresol_y : 2 * nresol_y]

# Pressure, Density and Temperature

p = np.zeros(nresol_z)
d = np.zeros(nresol_z)
t = np.zeros(nresol_z)

for iz in range(0, nresol_z):
    z = Z[iz]
    p[iz] = pres(ix, iy, iz, z, z0, deltaz, a, b, beta0, Bz, H, T0, T1)
    d[iz] = den(
        ix,
        iy,
        iz,
        z,
        z0,
        deltaz,
        a,
        b,
        Bx,
        By,
        Bz,
        dBzd,
        beta0,
        H,
        T0,
        T1,
        T_photosphere,
    )
    t[iz] = temp(
        ix,
        iy,
        iz,
        z,
        z0,
        deltaz,
        a,
        b,
        Bx,
        By,
        Bz,
        dBzd,
        beta0,
        H,
        T0,
        T1,
        T_photosphere,
    )

plt.plot(p, Z, label="Pressure")
plt.plot(d, Z, label="Density")
plt.plot(t, Z, label="Temperature")
plt.legend()
plt.show()
exit()

p = np.zeros((nresol_y, nresol_x, nresol_z))
d = np.zeros((nresol_y, nresol_x, nresol_z))
t = np.zeros((nresol_y, nresol_x, nresol_z))

for iz in range(0, nresol_z):
    for ix in range(0, nresol_x):
        for iy in range(0, nresol_y):
            x = x_arr[ix]
            y = y_arr[iy]
            z = Z[iz]
            p[iy, ix, iz] = plasma_parameters.pres(
                ix, iy, iz, z, z0, deltaz, a, b, beta0, Bz, H, T0, T1
            )
            d[iy, ix, iz] = plasma_parameters.den(
                ix,
                iy,
                iz,
                z,
                z0,
                deltaz,
                a,
                b,
                Bx,
                By,
                Bz,
                dBzd,
                beta0,
                H,
                T0,
                T1,
                T_photosphere,
            )
            t[iy, ix, iz] = plasma_parameters.temp(
                ix,
                iy,
                iz,
                z,
                z0,
                deltaz,
                a,
                b,
                Bx,
                By,
                Bz,
                dBzd,
                beta0,
                H,
                T0,
                T1,
                T_photosphere,
            )
