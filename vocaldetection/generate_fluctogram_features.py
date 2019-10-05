from __future__ import division
import numpy as np
import utils
import os

AUDIO_PATH = os.environ["AUDIO_PATH"]
FEAT_PATH = os.environ["FEAT_PATH"]

# Make 1 second summarization as features with half second of hop length
# 172 frames == 1 second (using 44100 samples per second)
feature_length = 96
half_sec = 48
samplerate = 44100

# Process files to create label and save on a given path
def generate_fluctogram(train_files):

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

            # get log freq axis
            bins_per_octave = 120
            target_freqs = librosa.cqt_frequencies(6*bins_per_octave, fmin=librosa.note_to_hz('E3'),
                                                   bins_per_octave=bins_per_octave)

            n_fft = 2048
            hop_length = 441
            y_stft = librosa.core.stft(audio, n_fft=n_fft, hop_length=hop_length)

            y_stft_log = fluctogram.stft_interp(y_stft, librosa.core.fft_frequencies(sr=sr, n_fft=n_fft), target_freqs)
            
            spec_flatness = spectral.bandwise_flatness(y_stft_log, target_freqs)
            spec_contraction = spectral.bandwise_contraction(y_stft_log, target_freqs)
            f = fluctogram.Fluctogram(y_stft_log, target_freqs)
            
            sf_shape = spec_flatness.shape[1]

            print("mfcc shape", int(sf_shape))

            #print (mfcc.shape)
            print("number of chunks", int(sf_shape/half_sec))

            feature_vector = []

            # For half second
            # Variance here is calculated based on 1 second related with the central frame
            feature_length = half_sec

            for chunk in range(int(sf_shape/half_sec)):
                start = chunk*half_sec
            
                if start < feature_length:
                    sf_var = np.std(spec_flatness[:,0:start+feature_length], 1)**2
                    sc_var = np.std(spec_contraction[:,0:start+feature_length], 1)**2
                    fl_var = np.std(f.fluctogram[:,0:start+feature_length], 1)**2
                else:
                    sf_var = np.std(spec_flatness[:,start-feature_length:start+feature_length], 1)**2
                    sc_var = np.std(spec_contraction[:,start-feature_length:start+feature_length], 1)**2
                    fl_var = np.std(f.fluctogram[:,start-feature_length:start+feature_length], 1)**2
                
                # Concatenate means and std. dev's into a single feature vector
                feature_vector.append(np.concatenate((sf_var, sc_var, fl_var),axis=0))
                #print("feature summary: {}".format(len(feature_vector)))

            # Store the feature vector and corresponding label in integer format
            for idx in range(len(feature_vector)):
                train_features.append(feature_vector[idx])
            print(" ")
            
            # Save Fluctogram and Indicators as npy file
            train_features = np.matrix(train_features)
            np.save(FEAT_PATH+os.path.basename(tf1)[:-8]+"_fluctogram.npy", train_features)


if __name__ == '__main__':
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import sys
    import json
    import spectral, fluctogram
    
    # Create a list of all musics
    music_files = []

    with open('vocal_pieces.json') as json_file:  
        data = json.load(json_file)
        
        for music in data.keys():
            music_files.append(AUDIO_PATH+music+"/"+music+"_MIX.wav")
    
    generate_fluctogram(music_files)
