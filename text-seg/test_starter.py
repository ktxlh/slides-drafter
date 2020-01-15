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

from nltk.tokenize import sent_tokenize
from tqdm import tqdm, trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset,
                              random_split)
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer, BertForTokenClassification,
                          get_linear_schedule_with_warmup)
from utils import remove_non_printable, traverse_json_dir

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# mini-json: Subset with only 7822*.json and 7823*.json
json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/" #TODO Unused. Comment it out

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)#, output_attentions=True,

############################################

tag2idx = {'B': 0, 'I': 1, 'O': 2}
tags_vals = ['B', 'I', 'O']
classifier_token = BertForTokenClassification.from_pretrained(model_dir, num_labels=len(tag2idx))



from utils import keywordextract

sentences = ["Some students space paragraphs, trying to separate points when the process of writing is over.","This is a major mistake.", "How much easier your writing would become if you looked at it from another angle!"]
sentence = sentences[0]
s0, s1, s2 = sentences[:3]
kw = keywordextract(s0, classifier_token, tokenizer)
kw = keywordextract(s1, classifier_token, tokenizer)
kw = keywordextract(s2, classifier_token, tokenizer)
