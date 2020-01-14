import os
from text_seg import traverse_json_dir
from nltk.tokenize import sent_tokenize

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
output_dir = json_dir+"-raw"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sections = traverse_json_dir(json_dir, toke_to_sent=False)

with open(os.path.join(output_dir,"raw.txt"),'w') as fout:
    fout.write("\n".join(sections)+"\n")