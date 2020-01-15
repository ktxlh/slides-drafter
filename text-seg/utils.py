import os
import json
from nltk.tokenize import sent_tokenize

def traverse_json_dir(json_dir, return_docs):

    def remove_non_printable(s):
        s = s.encode('ascii', errors='ignore').decode('ascii')
        s = '\n'.join([ss for ss in s.split('\n') if len(ss) > 40])
        return s

    rtn = []
    for root, dirs, files in os.walk(json_dir):
        print("# of json files in total:",len(files))
        files.sort()
        for fname in files:
            obj = json.load(open(os.path.join(json_dir, fname)))
            sections = []
            for secs in obj['now']['sections']:
                text = remove_non_printable(secs['text'])
                if len (text) > 40:
                    sentences = sent_tokenize(text)
                    sections.append(sentences)
            if return_docs:
                rtn.append(sections)
            else:
                rtn.extend(sections)
            
    print("# of sections loaded:", len(sections))
    return rtn