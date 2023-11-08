from load.read_file import read_fits_SOAR
from model.field.bfield_model import magnetic_field, bz_partial_derivatives
import numpy as np
import matplotlib.pyplot as plt
from model.plasma_parameters import deltapres, deltaden


path_blos: str = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)
data = read_fits_SOAR(path_blos)

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

deltaz: np.float64 = np.float64(z0 / 10.0)
# Background atmosphere

g: float = 272.2  # solar gravitational acceleration m/s^-2 gravitational acceleration

bfield: np.ndarray[np.float64, np.dtype[np.float64]] = magnetic_field(
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

dpartial_bfield: np.ndarray[np.float64, np.dtype[np.float64]] = bz_partial_derivatives(
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

b_back: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros((nresol_y, nresol_x))
b_back: np.ndarray[np.float64, np.dtype[np.float64]] = bfield[
    nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, 0, 2
]
maxcoord = np.unravel_index(np.argmax(b_back, axis=None), b_back.shape)

iy: int = int(maxcoord[0])
ix: int = int(maxcoord[1])

x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
)
y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
)
z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
)

# Background Pressure, Density and Temperature

delta_p: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(nresol_z)
delta_d: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(nresol_z)

for iz in range(0, nresol_z):
    z: np.float64 = z_arr[iz]
    bz: np.float64 = bfield[iy, ix, iz, 2]
    bzdotgradbz: np.float64 = (
        bfield[iy, ix, iz, 1] * dpartial_bfield[iy, ix, iz, 1]
        + bfield[iy, ix, iz, 0] * dpartial_bfield[iy, ix, iz, 0]
        + bfield[iy, ix, iz, 2] * dpartial_bfield[iy, ix, iz, 2]
    )
    delta_p[iz] = deltapres(z, z0, deltaz, a, b, bz)
    delta_d[iz] = deltaden(z, z0, deltaz, a, b, bz, bzdotgradbz, g)

plt.plot(z_arr, delta_p, label="Background pressure", linewidth=0.5)
plt.plot(z_arr, delta_d, label="Background density", linewidth=0.5)

plt.legend()
plt.show()
