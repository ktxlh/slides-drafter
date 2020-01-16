"""
For interactive python experiments
"""
import argparse
import json
import os
import random
import re
import string
from itertools import combinations

import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm, trange

import torch
from pytorch_pretrained_bert import BertForTokenClassification
from pytorch_pretrained_bert import BertTokenizer as pytTokenizer
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset,
                              random_split)
from transformers import AdamW, BertForSequenceClassification
from transformers import BertTokenizer as traTokenizer
from transformers import get_linear_schedule_with_warmup
from utils import remove_non_printable, traverse_json_dir

from text_seg import TextSplitter

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# mini-json: Subset with only 7822*.json and 7823*.json
json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/"
tok_model_path = "/home/shanglinghsu/BERT-Keyword-Extractor/model.pt"

tag2idx = {'B': 0, 'I': 1, 'O': 2}
tags_vals = ['B', 'I', 'O']

splitter = TextSplitter(model_dir, tok_model_path) 

############################################


text = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet."""
sentences = sent_tokenize(text)
sentence = sentences[0]
s0, s1, s2 = sentences[:3]

t0 = "Some students space paragraphs, trying to separate points when the process of writing is over."
t1 = "Some students space paragraphs"
t2 = "trying to separate points when the process of writing is over."
ts = [t0,t1,t2]

for sent in sentences:
    print(sent)
    kw = splitter.extract_keywords(sent)
    if len(kw) > 0:
        print(">>",kw)
    print()

for t in ts:
    print(t)
    kw = splitter.extract_keywords(t)
    if len(kw) > 0:
        print(">>",kw)
    print()

title, segments, keywords, subtitles = splitter.split(text)

######################
# Key: shorter senteces work better
"""
python keyword-extractor.py --path "model.pt" --sentence "The IOB format (short for inside, outside, beginning) is a common tagging format for tagging tokens in a chunking task in computational linguistics (ex. named-entity recognition)."

python keyword-extractor.py --path "model.pt" --sentence "Some students space paragraphs"

python keyword-extractor.py --path "model.pt" --sentence "trying to separate points when the process of writing is over."

python keyword-extractor.py --path "model.pt" --sentence "Such paragraphs demand a certain knowledge."
"""
"""
import test_starter
"""
