import os

from nltk.tokenize import sent_tokenize

from utils import traverse_json_dir

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
output_dir = json_dir+"-raw"
print("output_dir:", output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

docs = traverse_json_dir(json_dir, return_docs=False)
l_tr = int(float(len(docs))*0.8)

with open(os.path.join(output_dir,"train.txt"),'w') as fout:
    fout.write("\n".join([
        "\n".join(sents)+"\n" for sents in docs[:l_tr]
    ])+"\n")

with open(os.path.join(output_dir,"test.txt"),'w') as fout:
    fout.write("\n".join([
        "\n".join(sents)+"\n" for sents in docs[l_tr:]
    ])+"\n")