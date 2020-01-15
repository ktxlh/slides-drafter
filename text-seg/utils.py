import os
import json
import torch
from nltk.tokenize import sent_tokenize

def remove_non_printable(s):
        s = s.encode('ascii', errors='ignore').decode('ascii')
        s = '\n'.join([ss for ss in s.split('\n') if len(ss) > 40])
        return s

def traverse_json_dir(json_dir, return_docs):
    rtn = []
    print("*** Traversing ***")
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
    print("--- Traversing done ---")
    return rtn

def format_attention(attention):
    """
    attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
    """
    # From https://github.com/jessevig/bertviz/blob/138381e83d33cf3e221bd540cc1c704b5c4af99e/bertviz/util.py#L3
    squeezed = []
    for layer_attention in attention:
        # num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)