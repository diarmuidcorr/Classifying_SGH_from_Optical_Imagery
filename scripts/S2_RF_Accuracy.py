# Python Script to validate a RF algorithm using predefined
# Training and Testing Data from Sentinel-2 data
# based on and developed from a classification script developed by Chris Holden
# http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
# and compiled by Diarmuid Corr, Lancaster University (d.corr@lancaster.ac.uk, 
# https://github.com/diarmuidcorr) as part of his PhD project.

#packages
import numpy as np # math and array handling
from matplotlib import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
# calculating measures for accuracy assessment
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import datetime
import joblib
from sklearn.utils import class_weight

# defining output filenames and other variables
# how many cores should be used?
# -1 -> all available cores
n_cores = 15

# what is the attributes name of your classes in the shape file (field name of the classes)?
attribute = 'Class'

# Change this for the appropriate path
path_to_restults = 'Results/'

# directory, where the all meta results should be saved:
results_txt = path_to_restults + 'S2_Accuracy.txt'

# path to predefined training data
path_to_arrays = ''

training_data_name = 'S2_Img_Array.npz' # npz is numpy zip file extension
label_data_name = 'S2_ROI_Array.npz' # npz is numpy zip file extension

rf_classifier_name = 'S2_RF_Optimized_CV'



#Model performance including "Out-of-Bag" (OOB) prediction score, 
#band importance and a prediction Confusion Matrix.
def Model_Diagnostics(RF_Value, X_Value, y_Value):
    # With our Random Forest model fit, we can check out the "Out-of-Bag" (OOB)
    # prediction score:
    print('--------------------------------', file=open(results_txt, "a"))
    print('TRAINING and RF Model Diagnostics:', file=open(results_txt, "a"))
    print('OOB prediction of accuracy is: {oob}%'.format(oob=RF_Value.oob_score_ * 100))
    print('OOB prediction of accuracy is: {oob}%'.format(oob=RF_Value.oob_score_ * 100),
          file=open(results_txt, "a"))


    # we can show the band importance:
    bands = range(1,num_bands+1)
    i=0
    band_names = ['BR_NDWI', 'GNIR_NDWI', 'NWI', 'NDSI_MNDWI', 'SWI', 
                  'NDGI', 'SAVI_mod', 'SI_mod', 'NDI','AWEI_SH', 'AWEI_NSH', 
                  'TC_wet', 'Aerosol', 'Blue', 'Green', 'Red', 'Red Edge 1',
                  'Red Edge 2', 'Red Edge 3', 'NIR', 'Water Vapour',
                  'SWIRCirrus', 'SWIR1', 'SWIR2', 'Red Edge 4']
    
    print('Band Importance:', file=open(results_txt, "a"))
    for b, imp in zip(bands, RF_Value.feature_importances_):
        print('Band {b} = {band_name} importance: {imp}'.format(b=b, imp=imp,
                                                                band_name = band_names[i]))
        print('Band {b} = {band_name} importance: {imp}'.format(b=b, imp=imp,
                                                                band_name = band_names[i]),
              file=open(results_txt, "a"))
        i+=1


    # Let's look at a crosstabulation to see the class confusion. 
    # To do so, we will import the Pandas library for some help:
    # Setup a dataframe -- just like R
    # Exception Handling because of possible Memory Error

    try:
        rf_pred = RF_Value.predict(X_Value)
        df = pd.DataFrame()
        df['truth'] = y_Value
        df['predict'] = rf_pred

    except MemoryError:
        print('Crosstab not available ')

    else:
        # Cross-tabulate predictions
        print(pd.crosstab(df['truth'], df['predict'], margins=True))
        print(pd.crosstab(df['truth'], df['predict'], margins=True), file=open(results_txt, "a")) 
        
    cm = confusion_matrix(y_Value,rf_pred)
    print(classification_report(y_Value, rf_pred,
                                target_names=['Not Water', 'Water']))
    print(accuracy_score(y_Value, rf_pred, normalize=True))
    print(cm)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True, fmt='g', cmap='GnBu')
    plt.xlabel('classes - predicted') 
    plt.ylabel('classes - truth')
    plt.show()


# loading data and preparing for assessing the Random Forest Algorithm.

#prepare results text file:
with open(results_txt, "a+") as file:
    today = datetime.date.today()
    file.write('------------------------------------------------------------------------------\n')
    file.write('Random Forest Classification:' + str(today))

# Img_Array.npz contains the spectral information of the 25 bands which
# load in the training data
with np.load(path_to_arrays + training_data_name) as img_Savez:
    img_Final = img_Savez['Img_Final']

# ROI_Array.npz contains the labels for all training data defined above
with np.load(path_to_arrays + label_data_name) as roi_Savez:
    roi_Final = roi_Savez['ROI_Final']

# Calculate the class weights for assessment.
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(roi_Final),
                                                  y=roi_Final)

class_weights = dict(enumerate(class_weights.flatten(), 1))

# Define X and y values
X = img_Final
y = roi_Final

# Load rf classifier
rf = joblib.load(rf_classifier_name) #Load the RF algorithm

num_rows, num_bands = img_Final.shape
row = num_rows
col = 1
band_number = num_bands

# Compute model diagnostics - written to results_txt
Model_Diagnostics(rf, X, y)
