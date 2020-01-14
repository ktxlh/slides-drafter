import os

from nltk.tokenize import sent_tokenize

from utils import traverse_json_dir

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
output_dir = json_dir+"-raw"
print("output_dir:", output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sections = traverse_json_dir(json_dir, toke_to_sent=False, limit_paragraphs=0)
l_tr = float(len(sections))*0.8

with open(os.path.join(output_dir,"train.txt"),'w') as fout:
    fout.write("\n".join(sections[:l_tr])+"\n")

with open(os.path.join(output_dir,"test.txt"),'w') as fout:
    fout.write("\n".join(sections[l_tr:])+"\n")