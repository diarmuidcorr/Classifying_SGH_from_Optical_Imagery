# Script used to calculate the depth of supraglacial hydrological features
# from the red band in Landsat-8 imagery.
# Functions find and download corresponding Red L8 tile from Google Cloud Storage,
# Delete all features <=1800m^2 (which are not connected to another feature by a vertex),
# and calculate the depth of each pixel identified as lake by RF methods.


import sys
import os
import glob, shutil
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio import Affine
from shapely.geometry import Point
from shapely.validation import make_valid
import geopandas as gpd
from shapely.geometry import mapping
from multiprocessing import Pool

try:
    import gdal, ogr
except:
    from osgeo import gdal, ogr

import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category = ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)

    

# Uses URL returned above to download the red band tile.
# RF output is cropped to the L8 outline and both saved
# to file. The names of these files are returned with the red band as a numpy array. 
def band_loader(rf_file, red_band_path, lake_temp_name, bm_path): #, shape_layer
    #lake_temp_name = outputs_path + tile + '_temp_lakes.tif'
    red_ras = rasterio.open(red_band_path) #open tile with Gdal or Rasterio
    SW_ras = rasterio.open(rf_file) #Surface water raster defined by RF algorithm (or other).
    
    kwargs = SW_ras.meta.copy() 
    kwargs.update({'crs': SW_ras.crs,
                   'transform': SW_ras.transform,
                   'width': SW_ras.width,
                   'height': SW_ras.height
                   })
    #bedmachine raster can't be utilised for L8 processing as L8 tiles don't have consistent
    #dimensions. Clip with converted shapefiles instead!
    #bedmachine raster clipped to L8 tile extent
    '''rf_file_gdal = gdal.Open(rf_file)
    mem_drv = gdal.GetDriverByName('GTiff')
    mem_raster_ = mem_drv.Create(bm_path, rf_file_gdal.RasterXSize,
                                 rf_file_gdal.RasterYSize,
                                 1, gdal.GDT_Float32) 
    mem_raster_.SetProjection(rf_file_gdal.GetProjection())
    mem_raster_.SetGeoTransform(rf_file_gdal.GetGeoTransform())
    mem_band = mem_raster_.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)
    
    att_ = 'ATTRIBUTE=gridcode'
    gdal.RasterizeLayer(mem_raster_, [1], shape_layer, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
    
    mem_raster_.FlushCache()
    rf_file_gdal = None'''

    #with rasterio.open(bm_path, 'r', **kwargs) as dst:
     #   bm_array = dst.read(1).astype('float32')

    
    red_array = red_ras.read(1).astype('float32')
    SW_array = SW_ras.read(1).astype('float32')
    #if bm_array.max() < 2:
     #   os.remove(bm_path)
    

    #remove false positives from outline of tiles,
    #this will be when red_array=-32768 (no data)
    SW_array = np.where(((red_array >= 0)), SW_array, 0)
    #remove classified areas outside bedmachine ice sheet mask
    #SW_array = np.where(((bm_array == 2)), SW_array, 0) 

    

    with rasterio.open(lake_temp_name, 'w', **kwargs) as dst:
        dst.write_band(1, SW_array)
        

    #red_ras = None
    SW_ras = None
    SW_array = None
    
    return red_array, red_ras


# RF output (which is an array) is converted to a polygon (shapefile) layer and saved to file.
#The polygon layer, output hydrology featurs from RF algorithm,
#is returned as a GeoDataFrame from GeoPandas.
def shp_conversion(rf_file, lake_temp_name, tile, outputs_path):
    mask = None
    with rasterio.Env():
        with rasterio.open(lake_temp_name) as src:
            image = src.read(1)
            results = ({'properties': {'raster_val': v}, 'geometry': s}
                       for i, (s,v) in enumerate(shapes(image, mask = mask,  
                                                        connectivity = 8,
                                                        transform = src.transform)))
            dst_crs = src.crs
    
    geoms = list(results)
    polygonized_lake_ras = gpd.GeoDataFrame.from_features(geoms, crs = dst_crs)
    polygonized_lake_ras = correct_geom(polygonized_lake_ras)
    #SW raster to polygon
    results = None
    geoms = None
    image = None
    return polygonized_lake_ras, dst_crs


def correct_geom(polygonized_lake_ras):
    polygonized_lake_ras.geometry = polygonized_lake_ras.apply(lambda row:
                                                               make_valid(row.geometry)
                                                               if not row.geometry.is_valid
                                                               else row.geometry, axis=1)
    
    return polygonized_lake_ras


# Non-lake and small features <= 200 m^2 are deleted by this function.
#Returns an updated GeoDataFrame of the hydrological features.
def small_feature_del(rf_file, tile, polygonized_lake_ras, dst_crs):
    #The following code works to clip the features to bedmachine layer,
    #but takes too long for the rollout!
    #bed_machine_path = path_to_bedmachine + 'BedMachine_Mask.shp'
    #bed_machine = gpd.read_file(bed_machine_path).to_crs(dst_crs)
    
    # Clip the data using GeoPandas clip
    #polygonized_lake_ras = gpd.clip(polygonized_lake_ras,
                              #bed_machine) #clip polygons using bedmachine layer

    # Ignore missing/empty geometries
    polygonized_lake_ras = polygonized_lake_ras[~polygonized_lake_ras.is_empty]
    #keep only entries with gridcode = 2
    polygonized_lake_ras = polygonized_lake_ras[polygonized_lake_ras.raster_val == 2]
    

    if len(polygonized_lake_ras.index) == 0:
        #If SW polygons exist, finish loop.
        #polygonized_lake_ras = None
        return polygonized_lake_ras #just for consistency

    
    polygonized_lake_ras["POLY_AREA"] = polygonized_lake_ras['geometry'].area
    #get areas for each polygon and remove polygons below 200m^2.
    #if area <100 something funny has happened in area calculations so that negative
    #areas are summed.
    polygonized_lake_ras = polygonized_lake_ras[(polygonized_lake_ras.POLY_AREA > 1800) |
                                                (polygonized_lake_ras.POLY_AREA < 900)]

    #If SW polygons exist, finish loop.
    if len(polygonized_lake_ras.index) == 0:
        #polygonized_lake_ras = None
        return polygonized_lake_ras # for consistency

    polygonized_lake_ras = polygonized_lake_ras.reset_index(drop=True)
    polygonized_lake_ras['FID_USE'] = polygonized_lake_ras.index
    # equate FID_USE field to FID, needed later

    SW_raster = None
    bed_machine = None
    
    return polygonized_lake_ras


# Used to convert a shapefile layer to a raster, which is saved to file.
#It is called in depth_calc below.
def raster_burn(in_ras_template, driver, out_ras_name, ras_dtype, in_shp_name,
                fill_val, attribute):
    grid_raster = gdal.Open(in_ras_template, gdal.GA_ReadOnly)
    #Open the L8 Grid raster in Gdal
    proj = grid_raster.GetProjection() #Find projection of the L8 Grid raster
    transform = grid_raster.GetGeoTransform() #Find transform of the L8 Grid raster

    width = grid_raster.RasterXSize
    height = grid_raster.RasterYSize
    
    #Creates BedMachine Shapefile as a Raster.
    mask_drv = gdal.GetDriverByName(driver) #Driver
    mask_raster_ = mask_drv.Create(out_ras_name, 
                                   width, height,
                                   1, ras_dtype) # Create the output BM mask as raster
    
    SW_mask = ogr.Open(in_shp_name) #Read in BedMachine data as a Shapefile
    mask_layer = SW_mask.GetLayer() #Reads in BedMachine data as a layer

    mask_raster_.SetProjection(proj) #Define projection for the BM mask raster
    mask_raster_.SetGeoTransform(transform) #Define transform for the BM mask raster

    #Band 1 of the raster to be used (only 1 band in total)
    mask_band = mask_raster_.GetRasterBand(1) 
    mask_band.Fill(fill_val) #Fill raster with 0
    mask_band.SetNoDataValue(fill_val)
    
    att_ = 'ATTRIBUTE='+attribute
    
    gdal.RasterizeLayer(mask_raster_,  # output to our new dataset
                        [1],  # output to our new dataset's first band
                        mask_layer,  # rasterize this layer
                        None, None,  # don't worry about transformations in same projection
                        [1],  # burn value 1
                        [att_])  # put raster values according to the 'gridcode'
    mask_raster_.FlushCache() ##saves to disk!!
    del mask_raster_, grid_raster, mask_band, SW_mask, mask_layer

    return

# create a buffer around the polygonized_lake_ras layer of 90 m (3 L8 pixels).
def buffer_calc(polygonized_lake_ras):
    lake_buffer = polygonized_lake_ras.copy()
    lake_buffer = lake_buffer.buffer(90, cap_style=3, join_style=2)

    lake_buffer = gpd.GeoDataFrame(gpd.GeoSeries(lake_buffer))
    lake_buffer = lake_buffer.rename(columns={0:'geometry'}).set_geometry('geometry')
    
    lake_buffer = gpd.overlay(lake_buffer, polygonized_lake_ras, how='difference',
                              keep_geom_type=False, make_valid=True)

    return lake_buffer



def depth_calc(rf_file, red_ras, results_path, tile, lake_polygon_name, outputs_path,
               red_array, polygonized_lake_ras, lake_buffer, ad_path, red_band_path,
               depth_path, lake_raster_name):

    # extract the geometry in GeoJSON format
    geoms_ = lake_buffer.geometry.values # list of shapely geometries
    
    ad_values = []
    for polygon in geoms_:
        # transform to GeJSON format
        geoms = [mapping(polygon)]
        # extract the raster values values within the polygon
        #The Ad_out result is a Numpy masked array
        Ad_out, out_transform = mask(red_ras, geoms, crop=True) 

        data = Ad_out[0] # extract the values of the masked array

        no_data = red_ras.nodata
        row, col = np.where(data != no_data) 

        ad_val = np.extract(data != no_data, data)
        T1 = out_transform * Affine.translation(15, 15) # reference the pixel centre
        rc2xy = lambda r, c: (c, r) * T1  
        #Creation of a new resulting GeoDataFrame with the col, row and elevation values

        ad_poly = gpd.GeoDataFrame({'col':col,'row':row,'Ad_Val':ad_val})
        # coordinate transformation
        ad_poly['x'] = ad_poly.apply(lambda row: rc2xy(row.row,row.col)[0], axis=1)
        ad_poly['y'] = ad_poly.apply(lambda row: rc2xy(row.row,row.col)[1], axis=1)
        # geometry
        ad_poly['geometry'] = ad_poly.apply(lambda row: Point(row['x'], row['y']), axis=1)
        ad_poly = ad_poly[ad_poly.Ad_Val != 0]
        ad_value = ad_poly["Ad_Val"].mean()
        ad_values.append(ad_value)

    lake_buffer['Red_Ad'] = ad_values
    polygonized_lake_ras['Red_Ad'] = lake_buffer['Red_Ad']
    
    polygonized_lake_ras.to_file(lake_polygon_name, driver='ESRI Shapefile')
    #needs to be saved here
    #so that the next function can be executed.

    raster_burn(red_band_path, 'gtiff', ad_path, gdal.GDT_Float32,
                lake_polygon_name, 0, 'Red_Ad')
    #open tile with Gdal or Rasterio
    Red_Ad_ras = gdal.Open(ad_path, gdal.GA_ReadOnly) 
    #Defines a path for red Ad value raster for depth calc
    Red_Ad = np.array(Red_Ad_ras.GetRasterBand(1).ReadAsArray())

    #Depth calcs done here.
    #These were supplied by Laura Melling from Smith and Baker (1981) and the
    #Assumption from Maritorena et al., 1994 that gives attenuation coefficient, g=2Kd.
    g_red = 0.815175
    np.seterr(invalid = 'warn')
    np.seterr(divide = 'ignore') 
    red_depth = np.float32(((np.log(Red_Ad) - np.log(red_array)) / g_red))
    red_depth = np.where((red_depth < 0) | (red_depth == np.isnan), 0, red_depth)

    driver = gdal.GetDriverByName('GTiff')
    cols = red_array.shape[0]
    rows = red_array.shape[1]
    

    outdata_depth = driver.Create(depth_path, rows, cols, 1, gdal.GDT_Float32)
    ##sets same geotransform as input
    outdata_depth.SetGeoTransform(Red_Ad_ras.GetGeoTransform())
    ##sets same projection as input
    outdata_depth.SetProjection(Red_Ad_ras.GetProjection())
    outdata_depth.GetRasterBand(1).WriteArray(red_depth)
    ##saves to disk!!
    outdata_depth.FlushCache() 

    raster_burn(red_band_path, 'gtiff', lake_raster_name, gdal.GDT_Float32,
                lake_polygon_name, -1, 'FID_USE')


    geoms_ = polygonized_lake_ras.geometry.values # list of shapely geometries
    sum_values = []
    max_values = []
    depth_ras = rasterio.open(depth_path)
    for polygon in geoms_:
        # transform to GeJSON format
        geoms = [mapping(polygon)]
        # extract the raster values values within the polygon
        #The Ad_out result is a Numpy masked array
        depth_out, out_transform = mask(depth_ras, geoms, crop=True) 

        no_data = depth_ras.nodata
        data = depth_out[0] # extract the values of the masked array
        # extract the row, columns of the valid values
        row, col = np.where(data != no_data) 
        depth_val = np.extract(data != no_data, data)

        T1 = out_transform * Affine.translation(15, 15) # reference the pixel centre
        rc2xy = lambda r, c: (c, r) * T1  
        #Creation of a new resulting GeoDataFrame with the col, row and elevation values

        depth_poly = gpd.GeoDataFrame({'col':col,'row':row,'Depth_Val':depth_val})
        # coordinate transformation
        depth_poly['x'] = depth_poly.apply(lambda row: rc2xy(row.row,row.col)[0], axis=1)
        depth_poly['y'] = depth_poly.apply(lambda row: rc2xy(row.row,row.col)[1], axis=1)
        # geometry
        depth_poly['geometry'] = depth_poly.apply(lambda row:
                                                  Point(row['x'], row['y']), axis=1)
        depth_poly = depth_poly[depth_poly.Depth_Val != 0]

        depth_value = depth_poly["Depth_Val"].sum()
        max_depth = depth_poly["Depth_Val"].max()

        sum_values.append(depth_value)
        max_values.append(max_depth)


    polygonized_lake_ras['SUM'] = sum_values
    polygonized_lake_ras['MAX'] = max_values
    polygonized_lake_ras['Volume'] = polygonized_lake_ras['SUM'] * 900
    polygonized_lake_ras.to_file(lake_polygon_name, driver='ESRI Shapefile')
    
    polygonized_lake_ras = None 
    max_values = None
    sum_values = None
    max_depth = None
    depth_value = None
    Red_Ad_ras = None
    Red_Ad = None
    g_red = None
    red_depth = None
    geoms_ = None
    depth_ras = None
    geoms = None
    depth_out = None
    out_transform = None
    no_data = None
    data = None
    row = None
    col = None
    depth_val = None
    T1 = None
    rc2xy = None
    depth_poly = None
    
    return

def func_rollout(rf_file):
    #results_path = '../../../../../../scratch/hpc/23/corrd/Results/'
    results_path = '../../Landsat8/Results/HEC/'
    tile = rf_file[-44:-4] #tile name is 60 Characters long, remove file extension '.tif'
    red_band_path = results_path + 'Red_Bands/' + tile + '_temp_red.tif'
    lake_temp_name = results_path + 'Red_Bands/' + tile + '_temp_rfs.tif'
    outputs_path = rf_file[:-44] + 'Outputs/'
    lake_polygon_name = outputs_path + 'SW_' + tile + ".shp"
    depth_path = outputs_path + "Depth_Rasters/Depths_" + tile + ".tif"
    ad_path = results_path + 'Red_Bands/' + tile + '_Ad_Ras.tif'
    lake_raster_name = outputs_path + 'SW_' + tile + ".tif"
    bm_path = results_path + tile + '_BM_Temp.tif'
    

    red_array, red_ras = band_loader(rf_file, red_band_path, lake_temp_name,
                                     bm_path) #, shape_layer
    

    polygonized_lake_ras, dst_crs = shp_conversion(rf_file, lake_temp_name,
                                                   tile, outputs_path)
    
    polygonized_lake_ras = small_feature_del(rf_file, tile, polygonized_lake_ras,
                                             dst_crs)
    #Exit function if there are no features!
    if polygonized_lake_ras.empty:
        print(tile, 'contains no supraglacial water features! Process terminates.')
        #os.remove(red_band_path)
        os.remove(lake_temp_name)
        #below only if files are backed up elsewhere
        #os.remove(rf_file)
        os.rename(rf_file, results_path + 'Done_RF/' + tile + '.tif')
        return
    #Exit function if there are too many features as this slows the process down too much!
    if len(polygonized_lake_ras.index) >= 100000:
        print(tile, 'contains', len(polygonized_lake_ras.index),
              'features. Thid is too many supraglacial water features! Process terminates.')
        return 

    lake_buffer = buffer_calc(polygonized_lake_ras)
    
    depth_calc(rf_file, red_ras, results_path, tile, lake_polygon_name, outputs_path,
               red_array, polygonized_lake_ras, lake_buffer, ad_path, red_band_path,
               depth_path, lake_raster_name)

    #remove temp_files
    #os.remove(red_band_path)
    os.remove(lake_temp_name)
    os.remove(ad_path)
    #below  only if files are backed up elsewhere
    #os.remove(rf_file)
    os.rename(rf_file, results_path + 'Done_RF/' + tile + '.tif')
    print(tile, 'Done')
    
    results_path = None
    tile = None
    red_band_path = None
    lake_temp_name = None
    outputs_path = None
    lake_polygon_name = None
    depth_path = None
    ad_path = None
    lake_raster_name = None
    red_array = None
    red_ras = None
    polygonized_lake_ras = None
    dst_crs = None
    lake_buffer = None

    return

def pool_function(rf_file_paths):
    pool = Pool(12)
    pool.map(func_rollout, rf_file_paths)
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    #Change for each melt season
    ms_year = '2017/'
    ms_results_path = '../../Landsat8/Results/HEC/' + ms_year
    rf_file_paths = glob.glob(ms_results_path + '2017*/LC*.tif')
    pool_function(rf_file_paths)
