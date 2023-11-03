import astropy
from astropy.io import fits
import sunpy_soar
from sunpy.net import Fido, attrs as a
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import plot_magnetogram


# TO DO
# Extract boundary magnetic field vector from Magnetogram
# Gar nicht mal so einfach

# Magnetic field B Line Of Sight character
path_blos = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)

with fits.open(path_blos) as data:
    data.info()
    image_data = fits.getdata(path_blos, ext=0)
    # print(image_data.shape)
    photo_data = image_data[500:1000, 400:1200]

plot_magnetogram.plot_magnetogram_boundary(photo_data, 800, 500)
