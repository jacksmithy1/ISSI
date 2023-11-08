import numpy as np


class Data3D:
    def __init__(
        self,
        data_x: np.ndarray[np.float64, np.dtype[np.float64]],
        data_y: np.ndarray[np.float64, np.dtype[np.float64]],
        data_z: np.ndarray[np.float64, np.dtype[np.float64]],
        nresol_x: int,
        nresol_y: int,
        nresol_z: int,
        pixelsize_x: np.float64,
        pixelsize_y: np.float64,
        pixelsize_z: np.float64,
        nf_max: int,
        xmin: np.float64,
        xmax: np.float64,
        ymin: np.float64,
        ymax: np.float64,
        zmin: np.float64,
        zmax: np.float64,
        z0: np.float64,
    ):
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.nresol_x = nresol_x
        self.nresol_y = nresol_y
        self.nresol_z = nresol_z
        self.pixelsize_x = pixelsize_x
        self.pixelsize_y = pixelsize_y
        self.pixelsize_z = pixelsize_z
        self.nf_max = nf_max
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.z0 = z0


class DataBz:
    def __init__(
        self,
        data_z: np.ndarray[np.float64, np.dtype[np.float64]],
        nresol_x: int,
        nresol_y: int,
        nresol_z: int,
        pixelsize_x: np.float64,
        pixelsize_y: np.float64,
        pixelsize_z: np.float64,
        nf_max: int,
        xmin: np.float64,
        xmax: np.float64,
        ymin: np.float64,
        ymax: np.float64,
        zmin: np.float64,
        zmax: np.float64,
        z0: np.float64,
    ):
        self.data_z = data_z
        self.nresol_x = nresol_x
        self.nresol_y = nresol_y
        self.nresol_z = nresol_z
        self.pixelsize_x = pixelsize_x
        self.pixelsize_y = pixelsize_y
        self.pixelsize_z = pixelsize_z
        self.nf_max = nf_max
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.z0 = z0
