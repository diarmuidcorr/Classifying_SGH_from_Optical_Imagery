# Classifying_SGH_from_Optical_Imagery
Guide to Optical Imagery and Machine Learning Classification of Surface Water on Ice

This document contains information on how to find, visualise, download and manipulate optical satellite imagery for the Sentinel-2 and Landsat mission sensors, and how to train, test and execute scripts for the classification of ponded water on the surface of ice and/or snow.

This document was compiled by Diarmuid Corr, Lancaster University (d.corr@lancaster.ac.uk, https://github.com/diarmuidcorr) and details the work developed for his PhD project.

The satellite imagery download script to fetch Landsat & Sentinel data was developed by Vasco Nunes (https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud).

The Random Forest scripts are based on a classification script developed by Chris Holden (http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html).

Any questions or concerns should be addressed to Diarmuid Corr (d.corr@lancaster.ac.uk).

## 1 - Sources to visualise and download satellite imagery:

  There are many options available to visualise satellite imagery depending on what you want to see, which sensor you wish to sample, and whether you want to download the data. This list is not exhaustive.
  
  ### NASA/USGS EarthExplorer:
    
    To quickly visualise Sentinel-2 and Landsat imagery for a location and date range use the NASA/USGS tool, EarthExplorer:
      
https://earthexplorer.usgs.gov/
https://eos.com/products/landviewer/
https://apps.sentinel-hub.com/sentinel-playground/
  
    EarthExplorer contains sensor data from many other sources, and is more useful for visualisation rather than downloading due to slow download speeds and difficulty in batch downloading. 
    Note: A free account is needed for this data. 
    Note: Some data is unavailable to view and download from this source, it is not known why. 
    Note: EOS landviewer limited to 10 free scenes a day


  ### ESA Copernicus Open Access Hub:
    To visualise and download Sentinel mission data, use ESA’s Copernicus Open Access Hub:
    
https://scihub.copernicus.eu/dhus/#/home 
    
    This is the source for Sentinel-1,2,3 data. However, ESA removes the data when over one year old. This is frustrating if you wish to visualise/download older data.
    
    Note: A free account is needed for this data. 

  ### Google Earth Engine:

    To visualise and adapt satellite data online, use Google Earth Engine:

https://earthengine.google.com/
    
    Google Earth Engine combines a multi-petabyte catalogue of satellite imagery and geospatial datasets with planetary-scale analysis capabilities. Scientists, researchers, and developers use Earth Engine to detect changes, map trends, and quantify differences on the Earth's surface. Earth Engine is now available for commercial use, and remains free for academic and research use.
    
    Note: A free account is needed for this data.

  ### Google Cloud Storage buckets:
    To download data without visualisation, use public Google Cloud Storage buckets:

https://console.cloud.google.com/storage/browser/gcp-public-data-landsat - Landsat level 1 collection only.
https://cloud.google.com/storage/docs/public-datasets/landsat - documentation.

https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2/tiles/ - the Sentinel-2 level 1 and 2 collections only.
https://cloud.google.com/storage/docs/public-datasets/sentinel-2 - documentation.
      
  The data may be downloaded directly from these storage buckets, although this may prove inefficient. To batch download data, make use of the satellite imagery download script to fetch Landsat & Sentinel data, developed by Vasco Nunes:
  
https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud

  A sample script using this package is available in the script named:
    
    satellite_imagery_download.py 
    
  Note: The FeLS package from above should be installed on a python environment before this script can be executed:
  
    pip install fels   (difficult to install with conda)
      
  Note: All files will be downloaded to the chosen output directory. A tile catalogue is downloaded, this can exceed 5-10 GB. 
  Note: For information on variables and arguments see the documentation here:

https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud/blob/master/README.md

## 2 - Create RGB/True Colour imagery from satellite imagery:

Optical satellite sensors typically contain multispectral data with bands in the visible, near infrared, and short wave infrared part of the electromagnetic spectrum. An RGB image, sometimes referred to as a truecolor image, is a useful way to display the information stored in each pixel in a way that humans can understand. The colour of each pixel is determined by the combination of the red, green, and blue intensities stored in each colour plane at the pixel's location. The precision with which a real-life image can be replicated has led to the commonly used term truecolor image. 

  A sample script to combine RGB images for a Sentinel-2 or Landsat-8 is given in the script named:
  
    RGB_S2_L8.py 
  
  Note: The following packages are required on a python environment before this script can be executed: rasterio, glob, os, matplotlib, numpy, math, shutil, multiprocessing.





    
