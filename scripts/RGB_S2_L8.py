#script used to create RGB/Truecolour imagery from downloaded optical
#satellite imagery for a chosen sensor.
#Compiled by Diarmuid Corr, d.corr@lancaster.ac.uk

import rasterio, glob, os
from matplotlib import pyplot as plt
import numpy as np
import math
import shutil
from multiprocessing import Pool

# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


#combine Red, Green and Blue bands using Sentinel-2 imagery:
def s2_rgb_calc(imagePath):
    band2Path = imagePath + '_B02.jp2'
    band3Path = imagePath + '_B03.jp2'
    band4Path = imagePath + '_B04.jp2'
    tile = str(band3Path)[-148:-88] #name of the tile, e.g.:
    #S2*_MSIL1C_YYYYMMDDTHHMMSS_N****_R***_T*****_YYYYMMDDTHHMMSS
    
# open rasters useing rasterio
    band3 = rasterio.open(band3Path) #green
    band2 = rasterio.open(band2Path) #blue
    band4 = rasterio.open(band4Path) #red
# read opened rasters as arrays of dtype float32
    blue = band2.read(1).astype('float32')
    green = band3.read(1).astype('float32')
    red = band4.read(1).astype('float32')
    
# Normalize band DN into 0.0 - 1.0 scale
    red_norm = normalize(red)
    green_norm = normalize(green)
    blue_norm = normalize(blue)

# Stack bands
    rgbstack = np.stack((red_norm, green_norm, blue_norm), axis = 2)

# Save stacked bands as png
    rgb = plt.imsave(outpath + 'rgb_' + tile + '.png', rgbstack)
    
# Open Png using Rasterio and set variables to save as tif
    dataset = rasterio.open(outpath + 'rgb_'+tile+'.png', 'r')
    bands = [1, 2, 3]
    data = dataset.read(bands)
    transform = band3.transform
    crs = band3.crs

# Save dataset as a GeoTiff
    with rasterio.open(outpath + 'rgb_' + tile + '.tif', 'w', driver='GTiff',
                   width = band3.width, height = band3.height,
                   count = 3, dtype = data.dtype, nodata = 0,
                   transform = transform, crs = crs) as dst:
        dst.write(data, indexes =bands)

    os.remove(outpath + 'rgb_' + tile + '.png') #remove the png if it is not desired
    return

#combine Red, Green and Blue bands using Landsat-8 imagery:
def l8_rgb_calc(imagePath):
# define paths to each band: Blue, Green and Red.
    band2Path = imagePath + '_B2.TIF'
    band3Path = imagePath + '_B3.TIF'
    band4Path = imagePath + '_B4.TIF'

    tile = str(band2Path)[-47:-7] #name of the tile, e.g.:
    #LC08_L1**_******_YYYYMMDD_YYYYMMDD_01_T*
    
# read in bands, carry out DN to TOA reflectance conversion
# This is needed for Landsat 8 to be consistent with TOAs in Sentinel-2, see:
# https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product 
    metadataPath = open(imagePath+'_MTL.txt', 'r')
    metadata = metadataPath.readlines()
    metadataPath.close()

    for line in metadata:
            if 'REFLECTANCE_MULT_BAND_2' in line:
                MULT_BAND_2 = line.split('=')[-1]
                MULT_BAND_2 = MULT_BAND_2.strip('=')
                MULT_BAND_2 = float(MULT_BAND_2)
            if 'REFLECTANCE_ADD_BAND_2' in line:
                ADD_BAND_2 = line.split('=')[-1]
                ADD_BAND_2 = ADD_BAND_2.strip('=')
                ADD_BAND_2 = float(ADD_BAND_2)

            if 'REFLECTANCE_MULT_BAND_3' in line:
                MULT_BAND_3 = line.split('=')[-1]
                MULT_BAND_3 = MULT_BAND_3.strip('=')
                MULT_BAND_3 = float(MULT_BAND_3)
            if 'REFLECTANCE_ADD_BAND_3' in line:
                ADD_BAND_3 = line.split('=')[-1]
                ADD_BAND_3 = ADD_BAND_3.strip('=')
                ADD_BAND_3 = float(ADD_BAND_3)

            if 'REFLECTANCE_MULT_BAND_4' in line:
                MULT_BAND_4 = line.split('=')[-1]
                MULT_BAND_4 = MULT_BAND_4.strip('=')
                MULT_BAND_4 = float(MULT_BAND_4)
            if 'REFLECTANCE_ADD_BAND_4' in line:
                ADD_BAND_4 = line.split('=')[-1]
                ADD_BAND_4 = ADD_BAND_4.strip('=')
                ADD_BAND_4 = float(ADD_BAND_4)

            if 'SUN_ELEVATION' in line:
                SUN_ELEVATION = line.split('=')[-1]
                SUN_ELEVATION = SUN_ELEVATION.strip('=')
                SUN_ELEVATION = float(SUN_ELEVATION)

# open rasters useing rasterio
    band2 = rasterio.open(band2Path) #blue
    band3 = rasterio.open(band3Path) #green
    band4 = rasterio.open(band4Path) #re
    print('bands read')

# read opened rasters as numpy arrays with dtype float32
    np.seterr(divide='ignore', invalid='ignore')
    blue = band2.read(1).astype('float32')
    green = band3.read(1).astype('float32')
    red = band4.read(1).astype('float32')

# Convert to TOA reflectance, see:
# https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
    blue = ((MULT_BAND_2*blue + ADD_BAND_2)/
            (math.sin(SUN_ELEVATION * math.pi / 180)))
    
    green = ((MULT_BAND_3*green + ADD_BAND_3)/
             (math.sin(SUN_ELEVATION * math.pi / 180)))
    
    red = ((MULT_BAND_4*red + ADD_BAND_4)/
           (math.sin(SUN_ELEVATION * math.pi / 180)))

# Normalize band DN into 0.0 - 1.0 scale
    red_norm = normalize(red)
    green_norm = normalize(green)
    blue_norm = normalize(blue)
    print('bands normalized')

# Stack bands
    rgbstack = np.stack((red_norm, green_norm, blue_norm), axis = 2)

# Save stacked bands as png
    rgb = plt.imsave(outpath + 'rgb_' + tile + '.png', rgbstack)
    
# Open Png using Rasterio and set variables to save as tif
    dataset = rasterio.open(outpath + 'rgb_' + tile + '.png', 'r')
    bands = [1, 2, 3]
    data = dataset.read(bands)
    transform = band3.transform
    crs = band3.crs

# Save dataset as a GeoTiff
    with rasterio.open(outpath + 'rgb_' + tile + '.tif', 'w', driver='GTiff',
                   width = band3.width, height = band3.height,
                   count = 3, dtype = data.dtype, nodata = 0,
                   transform = transform, crs = crs) as dst:
        dst.write(data, indexes =bands)

    os.remove(outpath + 'rgb_' + tile + '.png') # remove the png if it is not desired
    return


def pool_function(imagePaths):
    pool = Pool() #open pool
# delete '#' as appropriate
    #pool.map(s2_rgb_calc, imagePaths) # function to run for Sentinel-2
    #pool.map(l8_rgb_calc, imagePaths) # function to run for Landsat-8
    pool.close()
    pool.join()
    return

if __name__ == '__main__':
    #define global variables:
    n_proc = 12 #example of the number of processors to be used.
    #Should not exceed number of CPUs on the server used.

    # delete ''' as appropriate
    #For Sentinel-2:
    '''path_to_tile_dir = '' # give the path to the directory where '*.SAFE' dirs are stored
    dir_list = glob.glob(path_to_tile_dir + 'S2*.SAFE')'''

    #For Landsat-8:
    '''path_to_tile_dir = '' # give the path to the directory where 'LC08*/' dirs are stored
    dir_list = glob.glob(path_to_tile_dir + 'LC08*')'''
    
    imagePaths = []
    # Delete ''' as required.
    #For Sentinel-2:
    '''for file in dir_list:
        # full path to Blue band of given tile
        path_to_band = file + '/GRANULE/L1C_*/IMG_DATA/*_B02.jp2'
        #add this path_to_band to imagePaths list with final 8 characters removed:
        #i.e.: '_B02.jp2'
        imagePaths.extend(glob.glob(path_to_band)[:-8])'''

    #For Landsat-8:
    '''for file in dir_list:
        # full path to Blue band of given tile
        path_to_band = file + 'LC08*_B2.TIF'
        #add this path_to_band to imagePaths list with final 8 characters removed:
        #i.e.: '_B02.jp2'
        imagePaths.extend(glob.glob(path_to_band)[:-7])''' 
        
    outpath = '' #set path to output RGB tifs too
    pool_function(imagePaths)
