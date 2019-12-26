import utils
import librosa as lr
import librosa
import pandas as pd
import numpy as np
import json
import os

# Path for source activations
SOURCEID_PATH = os.environ["MEDLEYDB_PATH"]+\
                "Annotations/Instrument_Activations/SOURCEID/"
AUDIO_PATH = os.environ["MEDLEYDB_PATH"]+"Audio/"
LABEL_PATH = os.environ["CODE_PATH"]+"Labels/"
FEAT_PATH = os.environ["CODE_PATH"]+"Features/"

# We've previously preprocessed our data and coverted all files 
# to a sample rate of 44100
samplerate = 44100

window_size = 4410
hop_size = 882
n_bands = 40
n_mfcc = 17
# Make 1 second summarization as features with half second of hop length
# 172 frames == 1 second (using 44100 samples per second)
#feature_length = 96
#half_sec = 48

# Specify the labels (classes) we're going to classify the data into
label0 = 'absent'
label1 = 'present'
labels = [label0, label1]


# Adjusted from MedleyDB API
VOCALS = ["male singer", "female singer", "vocalists"]
         #"male speaker", "female speaker",
         #"male rapper", "female rapper", "beatboxing"]


# Process files to create label and save on a given path
def generate_mfcc(train_files):

    if not os.path.exists(FEAT_PATH):
        os.makedirs(FEAT_PATH)
       

    # For every audio file in the training set, load the file, compute MFCCs, summarize them over time
    # using the mean and standard deviation (for each MFCC coefficient), and then save the features
    # and corresponding label in the designated lists and files
    for tf in train_files:
        
        # Define lists to store the training features and corresponding training labels
        train_features = []
        train_labels = []

        novocal = False

        print("filename: {:s}".format(os.path.basename(tf)))

        #mixes_path = '../MIXES/'+os.path.basename(tf)[:-8]+'/'

        for mix in [tf]: #glob.glob(os.path.join(mixes_path, '*.wav'))+[tf]: # This plus is to add the complete version
            print (mix)

            if 'novocal' in mix:
                print ('No vocal label')
                novocal = True
                continue

            if 'electronics' in mix or 'percussion' in mix or 'plucked' in mix or 'piano' in mix:
                continue

            tf1 = mix

            # Load audio
            audio, sr = librosa.load(tf1, sr=samplerate, mono=True)

            # Extract mfcc coefficients (remember we will discard the first one)
            # To see all the relevant kwarg arugments consult the documentation for
            # librosa.feature.mfcc, librosa.feature.melspectrogram and librosa.filters.mel
            mfcc = librosa.feature.mfcc(audio, sr=sr, n_fft=window_size, hop_length=hop_size,
                                        fmax=samplerate/2, n_mels=n_bands, n_mfcc=(n_mfcc + 1))

            # Discard the first coefficient
            #mfcc = mfcc[1:,:]

            print("mfcc shape", int(mfcc.shape[1]))

            # Read labels for each frame
            f0line = pd.read_csv(LABEL_PATH+os.path.basename(tf1)[:-8]+"_vocal.csv",index_col=None, header=None)
            f0line = f0line.values
            f0line = f0line.T[0]
            #timestamp = pd.DataFrame.as_matrix(f0line)[:,0]

            #print (mfcc.shape)
            #print("number of chunks", int(mfcc.shape[1]/half_sec))

            feature_vector = []
            tf_label = []

            # Delta features 
            mfcc_delta = librosa.feature.delta(mfcc)
            #mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            print (mfcc.shape,mfcc_delta.shape)
            for idx in range(mfcc.shape[1]):
                #print (idx)
                #print (mfcc[:,idx].shape)
                train_features.append(np.concatenate((mfcc[:,idx], mfcc_delta[:,idx])))
            #print (len(train_features))
            np.save(FEAT_PATH+os.path.basename(tf1)[:-8]+"_mfccedelta.npy", train_labels)
            """

            # For half second
            for chunk in range(int(mfcc.shape[1]/half_sec)):
                start = chunk*half_sec
                mfcc_means = np.mean(mfcc[:,start:start+feature_length], 1)
                mfcc_stddevs = np.std(mfcc[:,start:start+feature_length], 1)
                mfcc_max = np.max(mfcc[:,start:start+feature_length], 1)
                mfcc_median = np.median(mfcc[:,start:start+feature_length], 1)
                mfcc_d1_means = np.mean(mfcc_delta[:,start:start+feature_length], 1)
                mfcc_d1_stddevs = np.std(mfcc_delta[:,start:start+feature_length], 1)
                mfcc_d2_means = np.mean(mfcc_delta2[:,start:start+feature_length], 1)
                mfcc_d2_stddevs = np.std(mfcc_delta2[:,start:start+feature_length], 1)


                # We could do the same for the delta features like this:
                # mfcc_d1_means = np.mean(np.diff(mfcc), 1)
                # mfcc_d1_stddevs = np.std(np.diff(mfcc), 1)

                # Concatenate means and std. dev's into a single feature vector
                feature_vector.append(np.concatenate((mfcc_means, mfcc_stddevs, mfcc_max, mfcc_median,\
                                                      mfcc_d1_means, mfcc_d1_stddevs, mfcc_d2_means, mfcc_d2_stddevs\
                                                     ), axis=0))
                #print("feature summary: {}".format(len(feature_vector)))

                # Adjust labels to our classes
                if not novocal and len([x for x in f0line[start:start+feature_length] if x > 0]) >= half_sec: # Get 50% of frames
                    tf_label.append('present')
                else:
                    tf_label.append('absent')

            #tf_label = ['present' if x > 0.0 else 'abscent' for x in f0line]

            # Get labels index
            tf_label_ind = [labels.index(lbl) for lbl in tf_label]
            print("file label size: {:d}".format(len(tf_label_ind)))

            # Store the feature vector and corresponding label in integer format
            for idx in range(len(feature_vector)):
                train_features.append(feature_vector[idx])
                train_labels.append(tf_label_ind[idx]) 
            print(" ")
            
            # Save MFCCs as npy file
            train_features = np.matrix(train_features)
            np.save(FEAT_PATH+os.path.basename(tf1)[:-8]+"_mfcc.npy", train_features)
                
            # Save labels as npy file
            train_labels = np.array(train_labels)
            np.save(FEAT_PATH+os.path.basename(tf1)[:-8]+"_labels.npy", train_labels)
            """
            

# Process files to create label and save on a given path
def generate_vocal_variance(train_files):

    if not os.path.exists(FEAT_PATH):
        os.makedirs(FEAT_PATH)
       

    # For every audio file in the training set, load the file, compute MFCCs, summarize them over time
    # using the mean and standard deviation (for each MFCC coefficient), and then save the features
    # and corresponding label in the designated lists and files
    for tf in train_files:
        
        # Define lists to store the training features and corresponding training labels
        train_features = []

        print("filename: {:s}".format(os.path.basename(tf)))

        #mixes_path = '../MIXES/'+os.path.basename(tf)[:-8]+'/'

        for mix in [tf]: #glob.glob(os.path.join(mixes_path, '*.wav'))+[tf]: # This plus is to add the complete version
            print (mix)

            if 'novocal' in mix:
                print ('No vocal label')
                novocal = True
                continue

            if 'electronics' in mix or 'percussion' in mix or 'plucked' in mix or 'piano' in mix:
                continue

            tf1 = mix

            # Load audio
            audio, sr = librosa.load(tf1, sr=samplerate, mono=True)

            # Extract mfcc coefficients (remember we will discard the first one)
            # To see all the relevant kwarg arugments consult the documentation for
            # librosa.feature.mfcc, librosa.feature.melspectrogram and librosa.filters.mel
            mfcc = librosa.feature.mfcc(audio, sr=sr, n_fft=window_size, hop_length=hop_size,
                                        fmax=samplerate/2, n_mels=n_bands, n_mfcc=6) # 5 first MFCCs

            # Discard the first coefficient
            mfcc = mfcc[1:,:]

            print("mfcc shape", int(mfcc.shape[1]))

            #print (mfcc.shape)
            print("number of chunks", int(mfcc.shape[1]/half_sec))

            feature_vector = []

            # Delta features 
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # For half second
            for chunk in range(int(mfcc.shape[1]/half_sec)):
                start = chunk*half_sec
                if start < feature_length:
                    mfcc_var = np.std(mfcc[:,0:start+feature_length], 1)**2
                else:
                    mfcc_var = np.std(mfcc[:,start-feature_length:start+feature_length], 1)**2
                
                # Concatenate means and std. dev's into a single feature vector
                feature_vector.append(mfcc_var)
                #print("feature summary: {}".format(len(feature_vector)))

            # Store the feature vector and corresponding label in integer format
            for idx in range(len(feature_vector)):
                train_features.append(feature_vector[idx])
            print(" ")
            
            # Save Vocal Variance as npy file
            train_features = np.matrix(train_features)
            np.save(FEAT_PATH+os.path.basename(tf1)[:-8]+"_vocalvar.npy", train_features)
        
        

if __name__ == "__main__":

    # Create a list of all musics
    music_files = []

    with open('vocal_pieces.json') as json_file:  
        data = json.load(json_file)
        
        for music in data.keys():
            music_files.append(AUDIO_PATH+music+"/"+music+"_MIX.wav")

    # Saving vocal labels 
    generate_mfcc(music_files)
    generate_vocal_variance(music_files)
