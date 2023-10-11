# Classifying_SGH_from_Optical_Imagery
Guide to Optical Imagery and Machine Learning Classification of Surface Water on Ice

This document contains information on how to find, visualise, download and manipulate optical satellite imagery for the Sentinel-2 and Landsat mission sensors, and how to train, test and execute scripts for the classification of ponded water on the surface of ice and/or snow.

This document was compiled by Diarmuid Corr, Lancaster University (d.corr@lancaster.ac.uk, https://github.com/diarmuidcorr) and details the work developed for his PhD project.

The satellite imagery [download script](https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud) to fetch Landsat & Sentinel data was developed by Vasco Nunes.

The Random Forest scripts are based on a [classification script](http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html) developed by Chris Holden.

Any questions or concerns should be addressed to Diarmuid Corr (d.corr@lancaster.ac.uk).

## 1 - Sources to visualise and download satellite imagery:

  There are many options available to visualise satellite imagery depending on what you want to see, which sensor you wish to sample, and whether you want to download the data. This list is not exhaustive.
  
  ### NASA/USGS EarthExplorer:
    
    To quickly visualise Sentinel-2 and Landsat imagery for a location and date range use the NASA/USGS tool, EarthExplorer:
      
[EarthExplorer](https://earthexplorer.usgs.gov/)
[EOS Landviewer](https://eos.com/products/landviewer/)
[Sentinel Playground](https://apps.sentinel-hub.com/sentinel-playground/)
  
    EarthExplorer contains sensor data from many other sources, and is more useful for visualisation rather than downloading due to slow download speeds and difficulty in batch downloading. 
    Note: A free account is needed for this data. 
    Note: Some data is unavailable to view and download from this source, it is not known why. 
    Note: EOS landviewer limited to 10 free scenes a day


  ### ESA Copernicus Open Access Hub:
    To visualise and download Sentinel mission data, use ESA’s Copernicus Open Access Hub:
    
[Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home )
    
    This is the source for Sentinel-1,2,3 data. However, ESA removes the data when over one year old. This is frustrating if you wish to visualise/download older data.
    
    Note: A free account is needed for this data. 

  ### Google Earth Engine:

    To visualise and adapt satellite data online, use Google Earth Engine:

[Google Earth Engine](https://earthengine.google.com/)
    
    Google Earth Engine combines a multi-petabyte catalogue of satellite imagery and geospatial datasets with planetary-scale analysis capabilities. Scientists, researchers, and developers use Earth Engine to detect changes, map trends, and quantify differences on the Earth's surface. Earth Engine is now available for commercial use, and remains free for academic and research use.
    
    Note: A free account is needed for this data.

  ### Google Cloud Storage buckets:
    To download data without visualisation, use public Google Cloud Storage buckets:

  https://console.cloud.google.com/storage/browser/gcp-public-data-landsat - Landsat level 1 collection only.
  [Landsat data | Cloud Storage](https://cloud.google.com/storage/docs/public-datasets/landsat) - documentation.
  
  https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2/tiles/ - the Sentinel-2 level 1 and 2 collections only.
  [Sentinel-2 data | Cloud Storage](https://cloud.google.com/storage/docs/public-datasets/sentinel-2) - documentation.
      
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

## 3 - Train machine learning (random forest) algorithm:

Random Forest is a supervised classification approach which incorporates pixel-based (rather than object-based) training data. Specifically, we use the [Random Forest (Brieman 2001)](http://link.springer.com/article/10.1023/A:1010933404324) ensemble decision tree algorithm by [Leo Breiman and Adele Cutler](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm). The Random Forest algorithm has become extremely popular in the field of remote sensing, as it is computationally inexpensive, flexible and resistant to overfitting.
To learn more about Random Forest see:

    [Breiman, Leo. 2001. Random Forests. Machine Learning 45-1: 5-32](http://link.springer.com/article/10.1023/A:1010933404324)
    [Wikipedia - Random Forest](http://en.wikipedia.org/wiki/Random_forest)
    [Breiman and Cutler's website](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#workings)

Random Forest algorithms take the ensemble result of a large number of decision trees (a forest of them). The random part of the name comes from the term bootstrap aggregating, or bagging, meaning that each tree within the forest is trained only on some subset of the full training dataset (the subset is determined by sampling with replacement). The elements of the training data for each tree that are left unseen are held out-of-bag for estimation of the accuracy. Randomness helps decide which feature input variables are seen at each node for each decision tree. Once all individual trees are fit to the random subset of the training data, using a random set of feature variables at each node, the ensemble of them all is used to give the final prediction.

Finally, Random Forest has some other benefits:

    It gives a measure of variable importance which relates how useful your input features (e.g. spectral bands) are in the classification.
    The out-of-bag samples in each tree can be used to validate each tree. Grouping these predicted accuracies across all trees can sometimes give you an unbiased estimate of the error rate, similar to doing cross-validation.
    Can be used for regressions, unsupervised clustering, or supervised classification.
    Available in many popular languages, including Python, R, and MATLAB.
    Free, open source, and efficient.

Training a Random Forest algorithm requires training data representing all features the algorithm may encounter. This training data should be complemented by representative labels, in this algorithm a binary: water - not-water labels (or 0s and 1s). 
These scripts compare inputs of the multiple hyperparameters used to train a Random Forest algorithm: Total number of decision trees within the forest (n_estimators); the function applied to determine the maximum number of features (max_features); the maximum depth any given tree may reach before making a decision (max_dapth); the minimum number of samples before a split in any given tree (min_samples_split); and the minimum number of samples before a leaf in any given tree (min_samples_leaf). Two plots are recorded for each variation. The first: AUC Score (Area Under Curve) versus the varied hyperparameter (for both Training and Validation datasets). The second: prediction time (s)  versus the varied hyperparameter (for Validation dataset). 
Using the information obtained from these graphs, the script then performs a cross validation approach to choose the most appropriate values to train the final algorithm. The final algorithm is trained and fitted to the training data before saving.

There are scripts for each of the sensors (S2 and L8) in the following files:

    S2_RF_Train.py
    L8_RF_Train.py

Note: Both scripts require predefined training data and labels for these in a numpy array. They save the algorithm as a .joblib file.
Note: The following packages are required on a python environment before these scripts can be executed: rasterio, scikit-learn, gdal, numpy, matplotlib, datetime, time, joblib.Some of these packages will exist depending on your Python/conda install.

## 4 - Test/validate machine learning (random forest) algorithm:

Validation of the methods is carried out when choosing the hyperparameters in the training of the algorithm, however, we also test the methods on unseen datasets to ensure the methods are trained thoroughly and neither over- or under- fitting. This step requires datasets similar to last: data representing all features the algorithm may encounter with corresponding labels. The fitted algorithm from the step before should also be saved and used in this step. 

The scripts load in the test datasets and fitted Random Forest algorithm. The algorithm is used to classify the test data and outputs are compared to the labelled data. The accuracy assessment contains a confusion matrix, from which precision, recall, F1 Score and overall accuracy are computed (Equations 1a-1d). Precision is the number of true positive results divided by the number of all positive results, including those not identified correctly. Recall is the number of true positive results divided by the number of all samples that should have been identified as positive. F1 Score is an overall assessment of the model’s accuracy, it is the harmonic mean of the precision and recall. Accuracy is the proportion of correct predictions made from the full dataset.

    Precision = TPTP + FP    (1a) 
    Recall = TPTP + FN    (1b)
    F1 Score = 2  Precision  Recall  Precision + Recall = 2  TP2  TP + FP + FN    (1c)
    Accuracy = All TrueAll Values = TP + TNTP + FP + TN + FN    (1d)

There are scripts for each of the sensors (S2 and L8) in the following files:

    S2_RF_Accuracy.py 
    L8_RF_Accuracy.py 
    
Note: Both scripts require predefined testing data and labels for these in a numpy array as well as a saved algorithm as a .joblib file.
Note: The following packages are required on a python environment before these scripts can be executed: numpy, matplotlib, pandas, seaborn, scikit-learn, datetime, joblib. Some of these packages will exist depending on your Python/conda install.

## 5 - Calculate the depth of supraglacial hydrological features:

The depth of supraglacial hydrological features are caclculated using a radiative transfer equation and the red band in Landsat-8 and Sentinel-2 optical imagery. 

**Radiative Transfer Equation for the Depth of Supraglacial Water**

$$Z = {ln (Ad − R∞) − ln (Rw − R∞) \over -g}$$

Where:
Z is the depth in metres.160
Ad is the reflectance of the ice uderlying the supraglacial water (bottom albedo).
R∞ is the reflectance of optically deep water.
Rw is the reflectance of the SGH pixel of interest.
g is the coefficient for loss of spectral radiance in the water column. 
g ≈ 2Kd (Kd is the diffuse attenuation coefficient of downward light, (Maritorena et al., 1994)).

These scripts find and download corresponding Red L8 and S2 tiles from Google Cloud Storage and calculate the depth of each pixel identified as water by RF methods. There are scripts for each of the sensors (S2 and L8) in the following files:

    S2_Depth_Retrieval.py 
    L8_Depth_Retrieval.py 




