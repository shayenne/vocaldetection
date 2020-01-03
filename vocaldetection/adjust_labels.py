import utils
import librosa as lr
import librosa
import pandas as pd
import numpy as np
import json
import os


LABEL_PATH = os.environ["CODE_PATH"]+"Labels/"
FEAT_PATH = os.environ["CODE_PATH"]+"Features/"

# Specify the labels (classes) we're going to classify the data into
label0 = 'absent'
label1 = 'present'
labels = [label0, label1]

def adjust_labels(train_files, winsize, hopsize):
    
    for tf in train_files:
        # Read labels for each frame
        f0line = pd.read_csv(LABEL_PATH+tf+"_vocal.csv",index_col=None, header=None)
        f0line = f0line.values
        f0line = f0line.T[0]

        
        tf_label = []


        for idx in range(int(len(f0line)/hopsize)):
            # Adjust labels to our classes
            if len([x for x in f0line[hopsize*idx:hopsize*idx+winsize] if x > 0]) == winsize: # Get 100% of frames
                tf_label.append(1)
            elif len([x for x in f0line[hopsize*idx:hopsize*idx+winsize] if x > 0]) > 20: # Avoid remove frames with 20ms
                tf_label.append(-1)
            else:
                tf_label.append(0)
    
        # Save labels as npy file
        train_labels = np.array(tf_label)
        np.save(FEAT_PATH+tf+"_labels_"+str(winsize)+"ms.npy", train_labels)
        print ('Saved labels with',hopsize,'ms hopsize for', tf)


if __name__ == "__main__":

    # Create a list of all musics
    music_files = []

    with open('vocal_pieces.json') as json_file:  
        data = json.load(json_file)

    # Saving vocal labels for features by Lehner 
    winsize = 20 # miliseconds
    hopsize = 20 # miliseconds
    adjust_labels(data.keys(), winsize, hopsize)
    
    # Saving vocal labels for features VGGish 
    winsize = 960 # miliseconds
    hopsize = 480 # miliseconds
    adjust_labels(data.keys(), winsize, hopsize)