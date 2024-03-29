import utils
import medleydb as mdb
import json

# Load all multitracks
mtrack_generator = mdb.load_all_multitracks()

all_tracks_id = [mtrack.track_id for mtrack in mtrack_generator]

# Create a dictionary with all vocal pieces and its sources
mtrack_generator = mdb.load_all_multitracks()

soma = 0
lista = []
dicionario = {}
for mtrack in mtrack_generator:
    if not mtrack.is_instrumental:
        soma+=1
        lista.append(mtrack.track_id)
        dicionario[mtrack.track_id] = mtrack.stem_instruments

# Save dictionary to a json file
with open('vocal_pieces.json', 'w') as json_file:
    json.dump(dicionario, json_file)

print (" >> Generated vocal_pieces.json!")
