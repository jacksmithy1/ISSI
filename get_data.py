import scipy
import numpy as np

def get_magnetogram(path):
    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    #data_bz = data['b2dz5'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns  
    #data_bx = data['b2dx5'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    #data_by = data['b2dy5'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    #scale_height = data['h3d']

    data_bz = data['b2dz'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns  
    data_bx = data['b2dx'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data['b2dy'] # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data['info_unit'])
    print(data['info_pixel'])
    #print(data['info_boundary'])
    print(data['info_array'])

    bx_xlen = data_bx.shape[1]
    bx_ylen = data_bx.shape[0]
    by_xlen = data_by.shape[1]
    by_ylen = data_bx.shape[0]
    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    if (bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen): 
        print('x lengths of data do not match') 
        raise ValueError
    if (bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen): 
        print('y lengths of data do not match') 
        raise ValueError
    else:
        nresol_x = bx_xlen # Data resolution in x direction
        nresol_y = bx_ylen # Data resolution in y direction

    #nresol_z = scale_height.shape[0]
    nresol_z = 652
  
    ymin = 0.0 # Minimum value of y in data length scale, not in Mm
    ymax = 1.0 # Maximum value of y in data length scale, not in Mm
    xmin = 0.0 # Minimum value of x in data length scale, not in Mm
    xmax = nresol_x/nresol_y # Maximum value of x in data length scale, not in Mm
    zmin = 0.0 # Minimum value of z in data length scale, not in Mm
    zmax = nresol_z/nresol_y # Maximum value of z in data length scale, not in Mm
    zmax = 0.85

    nf_max = min(nresol_x, nresol_y, nresol_z)-1

    pixelsize_x = abs(xmax-xmin)/nresol_x # Data pixel size in x direction
    pixelsize_y = abs(ymax-ymin)/nresol_y # Data pixel size in y direction
    pixelsize_z = abs(zmax-zmin)/nresol_z # Data pixel size in z direction

    return [data_bx, data_by, data_bz, nresol_x, nresol_y, nresol_z, pixelsize_x, pixelsize_y, pixelsize_z, nf_max, xmin, xmax, ymin, ymax, zmin, zmax]

