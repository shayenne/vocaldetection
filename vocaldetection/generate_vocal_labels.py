import utils
import librosa as lr
import pandas as pd
import numpy as np
import json
import os

"""


"""

# Path for source activations
SOURCEID_PATH = os.environ["MEDLEYDB_PATH"]+\
		"Annotations/Instrument_Activations/SOURCEID/"
AUDIO_PATH = os.environ["MEDLEYDB_PATH"]+"Audio/"
LABEL_PATH = os.environ["CODE_PATH"]+"Labels/"


# Same from MedleyDB API
VOCALS = ["male singer", "female singer", "male speaker", "female speaker",
          "male rapper", "female rapper", "beatboxing", "vocalists"]


# Process files to create label and save on a given path (adjusted for VGGish size of frame)
def save_labels(files):

    if not os.path.exists(LABEL_PATH):
        os.makedirs(LABEL_PATH)
        
    # Process musics
    for music in files:
        source_activation = pd.read_csv(SOURCEID_PATH + music +\
                                        "_SOURCEID.lab", index_col=None)

        y, sr = lr.load(AUDIO_PATH + music + "/" + music + "_MIX.wav", sr=None)
        duration = lr.get_duration(y, sr)

        # Get music duration in miliseconds (1000 transform into miliseconds)
        label_vector = np.zeros(int(duration*1000)) 

        for idx, source in source_activation.iterrows():
            
            if source.instrument_label in VOCALS:
                start, end = source.start_time, source.end_time
                label_vector[int(start*1000):int(end*1000)] = 1
                
        df = pd.DataFrame(label_vector.astype('int').T,columns=None,index=None)

        # Save vocal labels and copy the audio file to the right place
        df.to_csv(LABEL_PATH + music+"_vocal.csv", index=False, header=False)
        #os.system('cp ' + AUDIO_PATH + music + '/' + music + '_MIX.wav' \
        #          + ' ' + dir_path)
        
        print ('Generated vocals file for', music)

    print (" >> Vocal labels completed.")
    


if __name__ == "__main__":

    # Create a list of all musics
    music_files = []

    with open('vocal_pieces.json') as json_file:  
        data = json.load(json_file)
        
        for music in data.keys():
            music_files.append(music)

    # Saving vocal labels 
    save_labels(music_files)
