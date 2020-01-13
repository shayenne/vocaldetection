import sys
sys.path.append('../vocaldetection/')
import sklearn
import utils
import json
import os
import pandas as pd
import numpy as np
#import seaborn as sns
import librosa
import numpy as np
from scipy.io import arff
#import matplotlib.pyplot as plt
#%matplotlib inline
import joblib


""" MFCC - LEHNER """
# Path for features calculated with Lehner Code
all_feat_path = '/media/shayenne/CompMusHD/DATASETS/MedleyDB/AllMusics/ICASSP2014/'
f = all_feat_path+'ICASSP2014RNN/'
mfcc_path = all_feat_path+'MFCC_29_30_0_0.5_0dt/40_20_40/'
# Read features and labels
FEAT_PATH = os.environ["FEAT_PATH"]
AUDIO_PATH = os.environ["AUDIO_PATH"]
PIECES = os.environ["PIECES_JSON"]

""" VGGISH """
# Read features and labels
VGGish_PATH = '/media/shayenne/CompMusHD/BRUCUTU/fasttmp/VGGISH/'
FEAT_PATH = os.environ["FEAT_PATH"]
PIECES = os.environ["PIECES_JSON"]

""" FEATURE SELECTED """
feature = 'VGGISH'

def __main__():

    music_files = []

    with open(PIECES) as json_file:  
        data = json.load(json_file)

        for music in data.keys():
            music_files.append(music)
    
    if feature == 'MFCC':
        X, y = read_mfcc_features(music_files)
    elif feature == 'VGGISH':
        X, y = read_vggish_features(music_files)
        
    # Percentage of singing voice frames on dataset
    print ('Percentage of singing voice frames on dataset:',round(sum(y)/len(X),3))
           
    nfolds = 2
           
    search_result = rf_param_selection(X, y, nfolds)
    
    filename = 'search_result_RF_VGGish.sav'
    #print (filename)
    joblib.dump(search_result, filename)
    
    filename = 'best_model_RF_VGGish.sav'
    #print (filename)
    joblib.dump(search_result.best_estimator_, filename)
    
    # Evaluate the estimator
    # Now lets predict the labels of the test data!
    predictions = search_result.best_estimator_.predict(X)
    
    from sklearn.metrics import precision_recall_fscore_support

    metrics = precision_recall_fscore_support(y, predictions)
    print('Negatives : ', metrics[3][0],'- Positives',metrics[3][1])
    print('Precision :', round(metrics[0][1],3))
    print('Recall    :', round(metrics[1][1],3))
    print('F-score   :', round(metrics[2][1],3))
    
    # lets compute the show the confusion matrix:
    cm = sklearn.metrics.confusion_matrix(y, predictions)
    print('Confusion Matrix:', cm)
    
    
    
def read_vggish_features(music_files):
    train_features = []
    train_labels = []

    for tf in music_files:
        # Load VGGish audio embeddings
        try:
            vggish = pd.read_csv(VGGish_PATH+os.path.basename(tf)+"_VGGish_PCA.csv",index_col=None, header=None)
            vggish = vggish.values

            print('.', end = '')
        except FileNotFoundError:
            print ('Não encontrei', os.path.basename(tf))
            continue

        
        lbl = np.load(FEAT_PATH+tf+"_labels_960ms.npy")
        #print (lbl.shape)

        feature_vector = []
        for idx in range(vggish.shape[0]):
            feature_vector.append(vggish[idx])

        # Store the feature vector and corresponding label in integer format
        for idx in range(len(feature_vector)):
            if lbl[idx] != -1: # Remove confusion frames
                train_features.append(feature_vector[idx])
                train_labels.append(lbl[idx])

    print ('\n> Load data completed!')
    return (np.array(train_features), np.array(train_labels))

        
def read_mfcc_features(music_files):
    train_features = []
    train_labels = []

    for tf in music_files:
        try:     
            dataset = arff.loadarff(mfcc_path+tf+'_MIX.arff')#arff.load(open(mfcc_path+tf+'_MIX.arff', 'r'))    
            data = pd.DataFrame(dataset[0]).values#np.array(dataset['data'])
            #print (mfcc_path+tf+'_MIX.arff')
            print ('.',end='')
        except FileNotFoundError:
            print ('File not found: ',mfcc_path+tf+'_MIX.arff')
            continue

        # Calculate VV because it is not included on Lehner feature pack
        #audiofile, _ = librosa.load(AUDIO_PATH+tf+'/'+tf+'_MIX.wav', sr=SR)
        #vv=lbl = np.load(FEAT_PATH+tf+"_vv_lee.npy")
        #print (vv.shape)
        lbl = np.load(FEAT_PATH+tf+"_labels_20ms.npy")

        feature_vector = []
        for idx in range(len(lbl)):
            #feature_vector.append(np.concatenate((data[idx], vv[idx]), axis=0))
            feature_vector.append(data[idx])

        # Store the feature vector and corresponding label in integer format
        for idx in range(len(feature_vector)):
            train_features.append(feature_vector[idx])
            train_labels.append(lbl[idx])

    print ('\n> Load data completed!')
    return (np.array(train_features), np.array(train_labels))
  

def rf_param_selection(X, y, nfolds):
    """ Search better hyperparameters for RF classifier
    
        This function receives data, target and number of
        folds for CV and returns an object with the grid
        search results of fitting all of the options.    
    """
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV       
    
    # Number of estimators to be considered
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 90, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    
    grid_search = GridSearchCV(rf, param_grid, cv = nfolds, 
                               verbose=10, n_jobs = 12)
    # Fit the grid search model
    grid_search.fit(X, y)
           
    return grid_search   
           

if __name__ == '__main__':
    __main__()