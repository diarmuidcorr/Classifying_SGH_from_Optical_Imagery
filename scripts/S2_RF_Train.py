
# Python Script to train a RF algorithm using predefined
# Training Data from Sentinel-2 data
# based on and developed from a classification script developed by Chris Holden
# http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
# and compiled by Diarmuid Corr, Lancaster University (d.corr@lancaster.ac.uk, 
# https://github.com/diarmuidcorr) as part of his PhD project.

#packages
from rasterio.transform import Affine
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np # math and array handling
from matplotlib import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestClassifier # classifier
from sklearn.model_selection import RandomizedSearchCV
import datetime
import time
import joblib



# defining output filenames and other variables
# how many cores should be used?
# -1 -> all available cores
n_cores = 15

# what is the attributes name of your classes in the shape file (field name of the classes)?
attribute = 'Class'

# Change this for the appropriate path
path_to_restults = 'Results/' 

# directory, where the all meta results should be saved:
results_txt = path_to_restults + 'S2_Training_Algorithm.txt'

# path to predefined training data
path_to_arrays = '' 

training_data_name = 'S2_Img_Array.npz' # npz is numpy zip file extension
label_data_name = 'S2_ROI_Array.npz' # npz is numpy zip file extension

rf_classifier_name = 'S2_RF_Optimized_CV'



###############################################################################################
###############################################################################################
#n_estimators = [100,200,300,400,500]
#Returns the max ROC score for the defined number of estimators (trees in RF)
def n_estimators_opt(estimators_range, X_train, y_train, X_test, y_test):
    for n_estimator in estimators_range:
        rf = RandomForestClassifier(n_estimators=n_estimator, n_jobs=n_cores,
                                    bootstrap=True, class_weight='balanced',
                                    criterion='gini', random_state=None, warm_start=False)
        rf.fit(X_train, y_train)
        start = time.time()
        train_pred = rf.predict(X_train)
        stop = time.time()
        duration = stop-start
        times.append(duration)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    return times, train_results, test_results

def est_graph_plotter(n_estimators):
    fig, ax = plt.subplots(1, 2)
    line1, = ax[0].plot(n_estimators, train_results, 'tab:blue', label="Train AUC")
    line2, = ax[0].plot(n_estimators, test_results, 'tab:red', label="Test AUC")
    line3, = ax[1].plot(n_estimators, times, 'tab:blue')
    ax[0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    ax[0].set_ylabel('AUC score')
    ax[0].set_xlabel('Number of Trees')
    ax[1].set_ylabel('Prediction Time (s)')
    ax[1].set_xlabel('Number of Trees')
    plt.tight_layout()
    plt.savefig(path_to_restults + 'TreeNoPlot.png')

    index = test_results.index(max(test_results))
    #test_results1 = test_results[:-10]
    #index = test_results1.index(max(test_results1))
    n_estimator = n_estimators[index]
    n_estimators = [n_estimator - 10, n_estimator - 5, n_estimator,
                    n_estimator + 5, n_estimator + 10]
    print('No. of trees: ', n_estimators, n_estimator)
    return n_estimators, n_estimator

#max_features = [None, 'sqrt', 'log2']
#max_features_label = ['None', 'sqrt()', 'log2()'] #Graph Labelling
#max_features_graph = [1,2,3] #Graph Labelling
#Returns the max ROC score for the defined max features function.
def max_features_opt(features_range, X_train, y_train, X_test, y_test):
    for max_feature in features_range:
        rf = RandomForestClassifier(n_estimators=n_estimator,
                                    max_features=max_feature, n_jobs=n_cores,
                                    bootstrap=True, class_weight='balanced',
                                    criterion='gini', random_state=None, warm_start=False)
        rf.fit(X_train, y_train)
        start = time.time()
        train_pred = rf.predict(X_train)
        stop = time.time()
        duration = stop-start
        times.append(duration)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    return times, train_results, test_results
    
def feature_plot(max_features, features_label, features_graph):
    max_features_label = features_label #Graph Labelling
    max_features_graph = features_graph #Graph functionality
    
    fig, ax = plt.subplots(1, 2)
    line1, = ax[0].plot(max_features_graph, train_results, 'tab:blue', label="Train AUC")
    line2, = ax[0].plot(max_features_graph, test_results, 'tab:red', label="Test AUC")
    line3, = ax[1].plot(max_features_graph, times, 'tab:blue')
    ax[0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    ax[0].set_ylabel('AUC score')
    ax[0].set_xlabel('Function on the No. of Features')
    ax[1].set_ylabel('Prediction Time (s)')
    ax[1].set_xlabel('Function on the No. of Features')
    #ax[0].set_xticks(max_features_graph,max_features_label)
    #ax[1].set_xticks(max_features_graph,max_features_label)
    ax[0].set_xticks(max_features_graph) # values
    ax[0].set_xticklabels(max_features_label) # labels
    ax[1].set_xticks(max_features_graph) # values
    ax[1].set_xticklabels(max_features_label) # labels
    plt.tight_layout()
    plt.savefig(path_to_restults + 'FeatureNoPlot.png')

    index = test_results.index(max(test_results))
    max_feature = max_features[index]
    max_features = [None, 'sqrt', 'log2']
    return max_features

#max_depths = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,None]
#max_labels = [11,'','','',15,'','','','',20,'','','','',25,'','','','',30,'','None']
#max_depths_graph = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
#Returns the max ROC score for the defined maximum tree depth.
def max_depths_opt(depths_range, X_train, y_train, X_test, y_test):
    for max_depth in depths_range:
        rf = RandomForestClassifier(n_estimators=n_estimator,
                                    max_depth=max_depth, n_jobs=n_cores,
                                    bootstrap=True, class_weight='balanced',
                                    criterion='gini', random_state=None, warm_start=False)
        rf.fit(X_train, y_train)
        start = time.time()
        train_pred = rf.predict(X_train)
        stop = time.time()
        duration = stop-start
        times.append(duration)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    return times, train_results, test_results

def depths_plot(max_depths, depths_labels, depth_graph):
    max_labels = depths_labels #Graph Labelling
    max_depths_graph = depth_graph #Graph functionality
    fig, ax = plt.subplots(1, 2)
    line1, = ax[0].plot(max_depths_graph, train_results, 'tab:blue', label="Train AUC")
    line2, = ax[0].plot(max_depths_graph, test_results, 'tab:red', label="Test AUC")
    line3, = ax[1].plot(max_depths_graph, times, 'tab:blue')
    ax[0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    ax[0].set_ylabel('AUC score')
    ax[0].set_xlabel('Max Depth of Trees')
    ax[1].set_ylabel('Prediction Time (s)')
    ax[1].set_xlabel('Max Depth of Trees')
    #ax[0].set_xticks(max_depths_graph,max_labels)
    #ax[1].set_xticks(max_depths_graph,max_labels)
    ax[0].set_xticks(max_depths_graph) # values
    ax[0].set_xticklabels(max_labels) # labels
    ax[1].set_xticks(max_depths_graph) # values
    ax[1].set_xticklabels(max_labels) # labels
    plt.tight_layout()
    plt.savefig(path_to_restults + 'MaxDepthPlot.png')

    index = test_results.index(max(test_results))
    max_depth = max_depths[index]
    max_depths = [max_depth - 5, max_depth, max_depth + 5, None]
    return max_depths

#min_samples_splits = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#Returns the max ROC score for the defined min samples split value.
def samples_splits_opt(samples_splits_range, X_train, y_train, X_test, y_test):
    for min_samples_split in samples_splits_range:
        rf = RandomForestClassifier(n_estimators=n_estimator,
                                    min_samples_split=min_samples_split, n_jobs=n_cores,
                                    bootstrap=True, class_weight='balanced',
                                    criterion='gini', random_state=None, warm_start=False)
        rf.fit(X_train, y_train)
        start = time.time()
        train_pred = rf.predict(X_train)
        stop = time.time()
        duration = stop-start
        times.append(duration)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    return times, train_results, test_results

def split_plot(min_samples_splits):
    fig, ax = plt.subplots(1, 2)
    line1, = ax[0].plot(min_samples_splits, train_results, 'tab:blue', label="Train AUC")
    line2, = ax[0].plot(min_samples_splits, test_results, 'tab:red', label="Test AUC")
    line3, = ax[1].plot(min_samples_splits, times, 'tab:blue')
    ax[0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    ax[0].set_ylabel('AUC score')
    ax[0].set_xlabel('Number of Samples (Nodes)')
    ax[1].set_ylabel('Predicition Time (s)')
    ax[1].set_xlabel('Number of Samples (Nodes)')
    plt.tight_layout()
    plt.savefig(path_to_restults + 'SamplesNodesPlot.png')

    index = test_results.index(max(test_results))
    min_samples_split = min_samples_splits[index]
    if min_samples_split == 2: #min possible value is 2.
        min_samples_splits = [2,3,4]
    else:
        min_samples_splits = [min_samples_split - 1, min_samples_split, min_samples_split + 1]
    return min_samples_splits

#min_samples_leafs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#Returns the max ROC score for the defined min samples leaf value.
def samples_leaf_opt(samples_leaf_range, X_train, y_train, X_test, y_test):
    for min_samples_leaf in samples_leaf_range:
        rf = RandomForestClassifier(n_estimators=n_estimator,
                                    min_samples_leaf=min_samples_leaf, n_jobs=n_cores,
                                    bootstrap=True, class_weight='balanced',
                                    criterion='gini', random_state=None, warm_start=False)
        rf.fit(X_train, y_train)
        start = time.time()
        train_pred = rf.predict(X_train)
        stop = time.time()
        duration = stop-start
        times.append(duration)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    return times, train_results, test_results

def leaf_plot(min_samples_leafs):
    fig, ax = plt.subplots(1, 2)
    line1, = ax[0].plot(min_samples_leafs, train_results, 'tab:blue', label="Train AUC")
    line2, = ax[0].plot(min_samples_leafs, test_results, 'tab:red', label="Test AUC")
    line3, = ax[1].plot(min_samples_leafs, times, 'tab:blue')
    ax[0].legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    ax[0].set_ylabel('AUC score')
    ax[0].set_xlabel('Number of Samples (Leaf)')
    ax[1].set_ylabel('Predicition Time (s)')
    ax[1].set_xlabel('Number of Samples (Leaf)')
    plt.tight_layout()
    plt.savefig(path_to_restults + 'SamplesLeafPlot.png')

    index = test_results.index(max(test_results))
    min_samples_leaf = min_samples_leafs[index]
    if min_samples_leaf == 1: #min possible value is 1.
        min_samples_leafs = [1,2,3]
    else:
        min_samples_leafs = [min_samples_leaf - 1, min_samples_leaf, min_samples_leaf + 1]
    return min_samples_leafs


#Ranges for each variable determined from the ROC training curves previously!
# Number of trees in random forest
#n_estimators = [200,300,400]  
# Number of features to consider at every split
#max_features = [None, 'sqrt', 'log2']
#max_depths = [20,25,30,None]
#min_samples_splits = [2,3,4]   
#min_samples_leafs = [1,2,3,4]
def rf_training(n_estimators, max_features, max_depths, min_samples_splits,
                min_samples_leafs, X, y):
    #Define the random grid to be used.
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depths,
                   'min_samples_split': min_samples_splits,
                   'min_samples_leaf': min_samples_leafs}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf_ = RandomForestClassifier(oob_score =True, verbose=1, n_jobs=1, bootstrap=True,
                                 class_weight='balanced', criterion='gini',
                                 random_state=None, warm_start=False)

    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf_, param_distributions = random_grid, 
                                   n_iter = 5000, cv = 3, verbose=1,
                                   random_state=42, n_jobs = n_cores)
    X = np.nan_to_num(X)
    rf = rf_random.fit(X, y)
    print(rf_random.best_params_)
    print('Best Parameters from Cross Validation: {}'.format(rf_random.best_params_),
          file=open(results_txt, "a"))
    
    return rf_random.best_params_

def mean_calc(x,y,z):
    mean_vals = []
    for i in range(len(x)):
        mean_val = (x[i] + y[i] + z[i])/3
        mean_vals.append(mean_val)
    return mean_vals


#All functions for preparing data, optimising, training and assessing the Random Forest Algorithm.

#prepare results text file:
with open(results_txt, "w+") as file:
    today = datetime.date.today()
    file.write('------------------------------------------------------------------------------')
    file.write('\nRandom Forest Classification started on:' + str(today))

# Img_Array.npz contains the spectral information of the 25 bands which
# load in the training data
with np.load(path_to_arrays + training_data_name) as img_Savez:
    img_Final = img_Savez['Img_Final']

# ROI_Array.npz contains the labels for all training data defined above
with np.load(path_to_arrays + label_data_name) as roi_Savez:
    roi_Final = roi_Savez['ROI_Final']


###############################################################################################
###############################################################################################
# Optimising Parameters
print('\nOptimising Parameters:', file=open(results_txt, "a"))
shape_img = img_Final #training values
num_rows, num_bands = shape_img.shape
row = num_rows
col = 1
band_number = num_bands

print('Image extent: {} x {} (row x col)'.format(row, col), file=open(results_txt, "a"))
print('Number of Bands: {}'.format(band_number), file=open(results_txt, "a"))

print('TRAINING', file=open(results_txt, "a"))

roi = roi_Final # roi values
X = shape_img[roi > 0, :] # slice to leave only the training values.
y = roi[roi > 0] # slice to leave only the training values.

print('Our X matrix is sized: {sz}'.format(sz=X.shape), file=open(results_txt, "a"))
print('Our y array is sized: {sz}'.format(sz=y.shape), file=open(results_txt, "a"))


num_water = y[y == 2]
num_not_water = y[y == 1]
print('There are {sz} samples in the Water dataset'.format(sz=num_water.shape),
      file=open(results_txt, "a"))

print('There are {sz} samples in the Not-Water dataset'.format(sz=num_not_water.shape),
      file=open(results_txt, "a"))

print('---------------------------------------', file=open(results_txt, "a"))


#Create binary thresholding for train_test_split functionality:
y_THRESHOLD = (y == 1) & (y != 0)
y[y_THRESHOLD] = 0
y_THRESHOLD = (y == 2) & (y != 0)
y[y_THRESHOLD] = 1

# define train and test datasets. Three of them allows for an average value to be calculated.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,stratify = y,
                                                    random_state=21)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.30,stratify = y,
                                                        random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.30,stratify = y,
                                                        random_state=2)

###############################################################################################
###############################################################################################
# Train a sample random forest varying number of trees = n_estimators (3 times)
train_results = []
test_results = []
times = []

# Define the number of trees to test
n_estimators = [2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,
                85,90,95,100,125,150,175,200,250,300,350,400,450,500]

# Test 1
times1, train_results1, test_results1 = n_estimators_opt(n_estimators,
                                                         X_train, y_train, X_test, y_test)
train_results = []
test_results = []
times = []

# Test 2
times2, train_results2, test_results2 = n_estimators_opt(n_estimators,
                                                         X_train1, y_train1, X_test1, y_test1)
train_results = []
test_results = []
times = []

# Test 3
times3, train_results3, test_results3 = n_estimators_opt(n_estimators,
                                                         X_train2, y_train2, X_test2, y_test2)

# calculate the means for time, train accuracy and test accuracy
times = mean_calc(times1,times2,times3)
train_results = mean_calc(train_results1,train_results2,train_results3)
test_results = mean_calc(test_results1,test_results2,test_results3)

# plot accuracy score for train and test datasets vs n_estimators
n_estimators, n_estimator = est_graph_plotter(n_estimators)

###############################################################################################
###############################################################################################
# Train a sample random forest varying the function applied to number of features (3 times)
train_results = []
test_results = []
times = []

# Define the max feature funtions to test over
max_features = [None, 'sqrt', 'log2']

# Labels and graph values to plot charts
max_features_label = ['None', 'sqrt()', 'log2()'] #Graph Labelling
max_features_graph = [1,2,3] #Graph Labelling

# Test 1
times1, train_results1, test_results1 = max_features_opt(max_features,
                                                         X_train, y_train, X_test, y_test)

# Test 2
train_results = []
test_results = []
times = []
times2, train_results2, test_results2 = max_features_opt(max_features,
                                                         X_train1, y_train1, X_test1, y_test1)

# Test 3
train_results = []
test_results = []
times = []
times3, train_results3, test_results3 = max_features_opt(max_features,
                                                         X_train2, y_train2, X_test2, y_test2)

# calculate the means for time, train accuracy and test accuracy
times = mean_calc(times1,times2,times3)
train_results = mean_calc(train_results1,train_results2,train_results3)
test_results = mean_calc(test_results1,test_results2,test_results3)

# plot accuracy score for train and test datasets vs max_features
max_features = feature_plot(max_features, max_features_label, max_features_graph)


###############################################################################################
###############################################################################################
# Train a sample random forest varying max depth of any given tree (3 times)
train_results = []
test_results = []
times = []

# Define the max depths to test over
max_depths = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
              23,24,25,26,27,28,29,30,31,None]

# Labels and graph values to plot charts
max_labels = [1,'','','',5,'','','','',10,'','','','',15,'','','','',20,'','',
              '','',25,'','','','','','','None']
max_depths_graph = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                    23,24,25,26,27,28,29,30,31,32]

# Test 1
times1, train_results1, test_results1 = max_depths_opt(max_depths,
                                                       X_train, y_train, X_test, y_test)

# Test 2
train_results = []
test_results = []
times = []
times2, train_results2, test_results2 = max_depths_opt(max_depths,
                                                       X_train1, y_train1, X_test1, y_test1)

# Test 3
train_results = []
test_results = []
times = []
times3, train_results3, test_results3 = max_depths_opt(max_depths,
                                                       X_train2, y_train2, X_test2, y_test2)

# calculate the means for time, train accuracy and test accuracy
times = mean_calc(times1,times2,times3)
train_results = mean_calc(train_results1,train_results2,train_results3)
test_results = mean_calc(test_results1,test_results2,test_results3)

# plot accuracy score for train and test datasets vs max_depths
max_depths = depths_plot(max_depths, max_labels, max_depths_graph)


###############################################################################################
###############################################################################################
# Train a sample random forest varying min samples before a split in any given tree (3 times)
train_results = []
test_results = []
times = []

# Define the  min samples before a split to test over
min_samples_splits = range(2, 101)

# Test 1
#Returns the max ROC score for the defined min samples split value.
times1, train_results1, test_results1 = samples_splits_opt(min_samples_splits,
                                                       X_train, y_train, X_test, y_test)

# Test 2
train_results = []
test_results = []
times = []
times2, train_results2, test_results2 = samples_splits_opt(min_samples_splits,
                                                       X_train1, y_train1, X_test1, y_test1)

# Test 3
train_results = []
test_results = []
times = []
times3, train_results3, test_results3 = samples_splits_opt(min_samples_splits,
                                                       X_train2, y_train2, X_test2, y_test2)


# calculate the means for time, train accuracy and test accuracy
times = mean_calc(times1,times2,times3)
train_results = mean_calc(train_results1,train_results2,train_results3)
test_results = mean_calc(test_results1,test_results2,test_results3)

# plot accuracy score for train and test datasets vs min_samples_splits
min_samples_splits = split_plot(min_samples_splits)


###############################################################################################
###############################################################################################
# Train a sample random forest varying min samples before a leaf in any given tree (3 times)
train_results = []
test_results = []
times = []

# Define the  min samples before a leaf to test over
min_samples_leafs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                     23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                     41,42,43,44,45,46,47,48,49,50]

# Test 1
times1, train_results1, test_results1 = samples_leaf_opt(min_samples_leafs,
                                                         X_train, y_train, X_test, y_test)

# Test 2
train_results = []
test_results = []
times = []
times2, train_results2, test_results2 = samples_leaf_opt(min_samples_leafs,
                                                         X_train1, y_train1, X_test1, y_test1)

# Test 2
train_results = []
test_results = []
times = []
times3, train_results3, test_results3 = samples_leaf_opt(min_samples_leafs,
                                                         X_train2, y_train2, X_test2, y_test2)

# calculate the means for time, train accuracy and test accuracy
times = mean_calc(times1,times2,times3)
train_results = mean_calc(train_results1,train_results2,train_results3)
test_results = mean_calc(test_results1,test_results2,test_results3)

# plot accuracy score for train and test datasets vs min_samples_leafs
min_samples_leafs = leaf_plot(min_samples_leafs)


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

# Example parameter values for the cross validation, values are returned from those above as well.
# Ensure the values from above 
'''
n_estimators = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,
                85,90,95,100,105,110,115,120,125]
max_depths = [5,6,7,8,9,10]
max_features = [None, 'sqrt', 'log2']
min_samples_splits = range(45, 65, 2)
min_samples_leafs = range(15, 30, 2)'''

# Function to carry out cross validation of the defined variables
best_param_ranges = rf_training(n_estimators, max_features, max_depths,
                                min_samples_splits, min_samples_leafs, X, y)


n_estimators = best_param_ranges['n_estimators']
min_samples_split = best_param_ranges['min_samples_split']
min_samples_leaf = best_param_ranges['min_samples_leaf']
max_features = best_param_ranges['max_features']
max_depth = best_param_ranges['max_depth']


#the following are the values used in this case from the above!
rf = RandomForestClassifier(n_estimators = n_estimators, max_depth=max_depth,
                            min_samples_split=min_samples_split, 
                            min_samples_leaf=min_samples_leaf, max_features=max_features,
                            oob_score =True, verbose=1, n_jobs=n_cores, bootstrap=True,
                            class_weight='balanced', criterion='gini',
                            random_state=None, warm_start=False)

X = np.nan_to_num(img_Final)
rf = rf.fit(img_Final, roi_Final)

# Save RF classifier
joblib.dump(rf, path_to_arrays + rf_classifier_name +'.joblib')
