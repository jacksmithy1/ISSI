import astropy
from astropy.io import fits
import sunpy_soar
from sunpy.net import Fido, attrs as a
from astropy.time import Time
import matplotlib.pyplot as plt

# Magnetic field B Line Of Sight character
path_blos = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)


with fits.open(path_blos) as data:
    data.info()
    image_data = fits.getdata(path_blos, ext=0)
    print(image_data.shape)
    plt.figure()
    photo_data = image_data[500:1000, 400:1200]
    plt.imshow(photo_data, cmap="bone")
    plt.colorbar()
    plt.show()
    # info = fits.getdata(path_blos, ext=10)

print(photo_data.shape)
# print(info)
