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
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
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

from utils import keywordextract

sentence = "Some students space paragraphs, trying to separate points when the process of writing is over."
kw = keywordextract(sentence, model, tokenizer)

