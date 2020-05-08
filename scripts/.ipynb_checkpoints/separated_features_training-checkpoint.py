import sys
sys.path.append('../vocaldetection/')
import sklearn
import utils
import json
import os
import pandas as pd
import numpy as np
#import seaborn as sns
#import librosa
import numpy as np
from scipy.io import arff
#import matplotlib.pyplot as plt
#%matplotlib inline
import joblib

dirname = os.path.dirname(__file__)


""" MFCC - LEHNER """
# Path for features calculated with Lehner Code
all_feat_path = '/media/DISCO2TB/datasets/MedleyDB/Features/ICASSP2014/'
f = all_feat_path+'ICASSP2014RNN/'
mfcc_path = all_feat_path+'MFCC_29_30_0_0.5_0dt/40_20_40/'

# Read features and labels
FEAT_PATH = '/media/DISCO2TB/datasets/MedleyDB/Features/'#os.environ["FEAT_PATH"]
AUDIO_PATH = os.environ["AUDIO_PATH"]
PIECES = os.environ["PIECES_JSON"]
PIECES_SPLIT =  os.path.join(dirname, 'split_train_test.json')

""" VGGISH """
# Read features and labels
VGGish_PATH = '/media/DISCO2TB/datasets/MedleyDB/VGGISH/'
#FEAT_PATH = os.environ["FEAT_PATH"]
#PIECES = os.environ["PIECES_JSON"]

""" FEATURE SELECTED """
feature = 'AGG'
metric_eval = 'f1'

def __main__():

    train_files = []
    test_files = []

    with open(PIECES_SPLIT) as json_file:  
        data = json.load(json_file)
    
        # Load train data
        for music in data['train']:
            train_files.append(music)
        #    print (music)
            
        # Load test data
        #print ('Test data')
        for music in data['test']:
            test_files.append(music)
        #    print (music)
    #sys.exit(0)
    
    
    if feature == 'MFCC':
        X_train, y_train = read_mfcc_features(train_files)
        X_test, y_test = read_mfcc_features(test_files)
    elif feature == 'VGGISH':
        X_train, y_train = read_vggish_features(train_files)
        X_test, y_test = read_vggish_features(test_files)
    elif feature == 'ALL':
        X_train, y_train = read_all_features(train_files)
        X_test, y_test = read_all_features(test_files)
    elif feature == 'AGG':
        X_train, y_train = read_agg_features(train_files)
        X_test, y_test = read_agg_features(test_files)    


    # Percentage of singing voice frames on dataset
    print ('Percentage of singing voice frames on dataset:')
    print ('For train:',round(sum(y_train)/len(X_train),3))
    print ('For test:',round(sum(y_test)/len(X_test),3))
    
    nfolds = 5
           
    search_result = rf_param_selection(X_train, y_train, nfolds)
    
    filename = 'search_result_RF_'+feature+'_'+metric_eval+'.sav'
    #print (filename)
    joblib.dump(search_result, filename)
    
    filename = 'best_model_RF_'+feature+'_'+metric_eval+'.sav'
    #print (filename)
    joblib.dump(search_result.best_estimator_, filename)

    print ('===== TRAIN EVALUATION ====')
    evaluate(search_result.best_estimator_, X_train, y_train)
    
    print ('===== TEST EVALUATION ====')
    evaluate(search_result.best_estimator_, X_test, y_test)
    
    
def evaluate(clf, X, y):
    from sklearn.metrics import precision_recall_fscore_support

    # Evaluate the estimator
    # Now lets predict the labels of the test data!
    predictions = clf.predict(X)
    
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
  
# TODO: arrumar esta função, ela ainda não faz o que deveria (ler fluctogram features)
def read_all_features(music_files):
    train_features = []
    train_labels = []

    for tf in music_files:
        try:     
            dataset = arff.loadarff(f+tf+'_MIX.arff')#arff.load(open(mfcc_path+tf+'_MIX.arff', 'r'))    
            data = pd.DataFrame(dataset[0]).values#np.array(dataset['data'])
            #print (mfcc_path+tf+'_MIX.arff')
            print ('.',end='')
        except FileNotFoundError:
            print ('File not found: ',f+tf+'_MIX.arff')
            continue

        # Calculate VV because it is not included on Lehner feature pack
        #audiofile, _ = librosa.load(AUDIO_PATH+tf+'/'+tf+'_MIX.wav', sr=SR)
        #vv=lbl = np.load(FEAT_PATH+tf+"_vv_lee.npy")
        #print (vv.shape)
        lbl = np.load(FEAT_PATH+tf+"_labels_20ms.npy")
        
        print (data.shape)

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

def read_agg_features(music_files):
    train_features = []
    train_labels = []

    for tf in music_files:
        try:     
            #dataset = arff.loadarff(FEAT_PATH+'AGG/'+tf+'_MIX.arff')#arff.load(open(mfcc_path+tf+'_MIX.arff', 'r'))    
            data = np.load(FEAT_PATH+'AGG/'+tf+"_agg.npy")
            #print (mfcc_path+tf+'_MIX.arff')
            print ('.',end='')
        except FileNotFoundError:
            print ('File not found: ',FEAT_PATH+'AGG/'+tf+'_MIX.arff')
            continue

        # Calculate VV because it is not included on Lehner feature pack
        #audiofile, _ = librosa.load(AUDIO_PATH+tf+'/'+tf+'_MIX.wav', sr=SR)
        #vv=lbl = np.load(FEAT_PATH+tf+"_vv_lee.npy")
        #print (vv.shape)
        lbl = np.load(FEAT_PATH+tf+"_labels_200ms.npy")
        
        #print (data.shape)

        feature_vector = []
        for idx in range(len(lbl)):
            feature_vector.append(data[idx])

        # Store the feature vector and corresponding label in integer format
        for idx in range(len(feature_vector)):
            if lbl[idx] != -1: # Remove confusion frames
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
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 110, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 30, num = 3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the parameters grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   #'min_samples_split': min_samples_split,
                   #'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    
    grid_search = GridSearchCV(rf, param_grid, cv = nfolds, 
                               scoring = metric_eval, verbose = 10, n_jobs = -1)
    # Fit the grid search model
    grid_search.fit(X, y)
           
    return grid_search   
           

if __name__ == '__main__':
    __main__()
