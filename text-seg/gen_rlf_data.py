import os

from nltk.tokenize import sent_tokenize

from utils import traverse_json_dir

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
output_dir = json_dir+"-raw"
print("output_dir:", output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sections = traverse_json_dir(json_dir, toke_to_sent=False, limit_paragraphs=0)

for i in range(3):
    print(sections[i])

with open(os.path.join(output_dir,"raw.txt"),'w') as fout:
    fout.write("\n".join(sections)+"\n")
