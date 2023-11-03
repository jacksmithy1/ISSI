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
# calculate z0 and delta z automatically from data
# optimise choice of a and alpha, switch case for potential, LFF and MHS
# automate nresol_z and zmax for generic data

a = 0.0
alpha = 0.0
    
b = 1.0
z0 = 0.4 # Where transition from NFF to FF takes place
deltaz = z0/10.0 # Width of transitional region

h1 = 0.0001 # Initial step length for fieldline3D
eps = 1.0e-8 # Tolerance to which we require point on field line known for fieldline3D
hmin = 0.0 # Minimum step length for fieldline3D
hmax = 1.0 # Maximum step length for fieldline3D

#data = get_data.get_magnetogram('Analytic_boundary_data.sav')

data = get_data.get_magnetogram('RMHD_boundary_data.sav')

#BFieldvec_Seehafer = np.load('field_data_potential.npy') 

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

#plot_magnetogram.plot_magnetogram_boundary(data_bz, nresol_x, nresol_y)

data_bz_Seehafer = Seehafer.mirror_magnetogram(data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y)

#plot_magnetogram.plot_magnetogram_boundary(data_bz_Seehafer, 2*nresol_x, 2*nresol_y)

#plot_magnetogram.plot_magnetogram_boundary_3D(data_bz_Seehafer, nresol_x, nresol_y, -xmax, xmax, -ymax, ymax, zmin, zmax)

B_Seehafer = BField_model.get_magnetic_field(data_bz_Seehafer, z0, deltaz, a, b, alpha, -xmax, xmax, -ymax, ymax, zmin, zmax, nresol_x, nresol_y, nresol_z, pixelsize_x, pixelsize_y, nf_max)

current_time = datetime.datetime.now()
dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

path = '/Users/lilli/Desktop/ISSI_data/field_data_' + str(a) + '_' + str(b) + '_' + str(alpha) + '_' + str(nf_max) + '_' + dt_string + '.npy'

with open(path, 'wb') as file:
    np.save(file, B_Seehafer)

#b_back_test = np.zeros((2*nresol_y, 2*nresol_x))
#b_back_test = B_Seehafer[:, :, 0, 2]
#plot_magnetogram.plot_magnetogram_boundary_3D(b_back_test, nresol_x, nresol_y, -xmax, xmax, -ymax, ymax, zmin, zmax)

plot_magnetogram.plot_fieldlines_grid(B_Seehafer, h1, hmin, hmax, eps, nresol_x, nresol_y, nresol_z, -xmax, xmax, -ymax, ymax, zmin, zmax)