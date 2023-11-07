from load.read_file import read_fits_SOAR
from model.field.bfield_model import magnetic_field, bz_partial_derivatives
import numpy as np
import matplotlib.pyplot as plt
from model.plasma_parameters import deltapres, deltaden


path_blos: str = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)
data = read_fits_SOAR(path_blos)

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

# Additional model parameters for tanh height profile
deltaz = z0 / 10.0  # Width of transitional region ca. 200km
a = 0.24
alpha = 0.5
b = 1.0

# Background atmosphere

rho0 = 2.7 * 10.0**-7.0  # g/m^3

g = 272.2  # solar gravitational acceleration m/s^-2 gravitational acceleration
mbar = 1.0  # photospheric mean molecular weight, as multiples of proton mass
B0 = 500.0  # normalising photospheric B-field in Gauss

t_photosphere = 5600.0  # temperature photosphere in Kelvin
t_corona = 2.0 * 10.0**6.0  # temperature corona in Kelvin
t0 = (t_photosphere + t_corona * np.tanh(z0 / deltaz)) / (1.0 + np.tanh(z0 / deltaz))
t1 = (t_corona - t_photosphere) / (1.0 + np.tanh(z0 / deltaz))
p0 = (
    1.3807 * t_photosphere * rho0 / (mbar * 1.6726) * 1.0
)  # photospheric plasma pressure in Pascal
pB0 = 3.9789 - 3 * B0**2.0  # photospheric magnetic pressure in Pascal
beta0 = p0 / pB0  # photospheric plasma beta
h = 1.3807 * t0 / (mbar * 1.6726 * g) * 0.001

t_init = np.array(
    [6000.0, 5500.0, 10000.0]
)  # Temperature profile from ISSI at heights h_init in Kelvin
h_init = np.array([0.0, 500.0, 2000.0])  # Heights for t_init in Mm

B = magnetic_field(
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

dBzd = bz_partial_derivatives(
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

b_back = np.zeros((nresol_y, nresol_x))
b_back = B[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, 0, 2]
maxcoord = np.unravel_index(np.argmax(b_back, axis=None), b_back.shape)

iy = maxcoord[0]
ix = maxcoord[1]

x_arr = np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
y_arr = np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

# Background Pressure, Density and Temperature
print(z_arr)
delta_p = np.zeros(nresol_z)
delta_d = np.zeros(nresol_z)

for iz in range(0, nresol_z):
    z = z_arr[iz]
    bz = B[iy, ix, iz, 2]
    bzdotgradbz = (
        B[iy, ix, iz, 1] * dBzd[iy, ix, iz, 1]
        + B[iy, ix, iz, 0] * dBzd[iy, ix, iz, 0]
        + B[iy, ix, iz, 2] * dBzd[iy, ix, iz, 2]
    )
    delta_p[iz] = deltapres(z, z0, deltaz, a, b, bz)
    delta_d[iz] = deltaden(z, z0, deltaz, a, b, bz, bzdotgradbz, g)

plt.plot(z_arr, delta_p, label="Background pressure", linewidth=0.5)
plt.plot(z_arr, delta_d, label="Background density", linewidth=0.5)

plt.legend()
plt.show()
