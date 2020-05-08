import os

os.environ["MEDLEYDB_PATH"] = "/media/shayenne/CompMusHD/DATASETS/MedleyDB/"

os.environ["CODE_PATH"] = "/home/shayenne/repositories/DATASETS/MedleyDB/"

# Path for source activations
os.environ["SOURCEID_PATH"] = os.environ["MEDLEYDB_PATH"]+\
                "Annotations/Instrument_Activations/SOURCEID/"
os.environ["AUDIO_PATH"] = os.environ["MEDLEYDB_PATH"]+"Audio/"
os.environ["LABEL_PATH"] = os.environ["CODE_PATH"]+"Labels/"
os.environ["FEAT_PATH"] = os.environ["CODE_PATH"]+"Features/"

os.environ["PIECES_JSON"] = "/home/shayenne/repositories/vocaldetection/vocaldetection/vocal_pieces.json"
