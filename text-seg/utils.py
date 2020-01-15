import os
import json
import torch
from nltk.tokenize import sent_tokenize
from tqdm import trange
import numpy as np

def remove_non_printable(s):
        s = s.encode('ascii', errors='ignore').decode('ascii')
        s = '\n'.join([ss for ss in s.split('\n') if len(ss) > 40])
        return s

def traverse_json_dir(json_dir, return_docs):
    rtn = []
    num_sentences_counter = 0
    print("*** Traversing ***")
    for root, dirs, files in os.walk(json_dir):
        print("# of json files in total:",len(files))
        files.sort()
        for i in trange(len(files)):
            fname = files[i]
            obj = json.load(open(os.path.join(json_dir, fname)))
            sections = []
            for secs in obj['now']['sections']:
                text = remove_non_printable(secs['text'])
                if len (text) > 40:
                    sentences = sent_tokenize(text)
                    num_sentences_counter += len(sentences)
                    sections.append(sentences)
            if return_docs:
                rtn.append(sections)
            else:
                rtn.extend(sections)
    print("# of sentences in total:",num_sentences_counter)
    print("--- Traversing done ---")
    return rtn
