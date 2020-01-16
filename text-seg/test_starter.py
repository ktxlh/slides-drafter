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
from pytorch_pretrained_bert import BertForTokenClassification
from text_seg import TextSplitter
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
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/"
tok_model_path = "/home/shanglinghsu/BERT-Keyword-Extractor/model.pt"

tag2idx = {'B': 0, 'I': 1, 'O': 2}
tags_vals = ['B', 'I', 'O']

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)#, output_attentions=True,
token_classifier = BertForTokenClassification.from_pretrained(tok_model_path, num_labels=len(tag2idx))
splitter = TextSplitter(model_dir) 

############################################


text = """Some students space paragraphs, trying to separate points when the process of writing is over. This is a major mistake. How much easier your writing would become if you looked at it from another angle! It is reasonable to use different types of paragraphs WHILE you are writing. In case you follow all the rules, you'll have no difficulty in bringing your message across to your reader.
If you browse for ‘the types of paragraphs' you'll be surprised how many results you'll get. Among others, the four following types should be distinguished: descriptive, expository, narrative, and persuasive paragraphs. Mastering these types will help you a lot in writing almost every type of texts.
Descriptive: These paragraphs have four main aims. First of all, they naturally describe something or somebody, that is conveying the information. Secondly, such paragraphs create powerful images in the reader's mind. Thirdly, they appeal to the primary senses of vision, hearing, touch, taste, and smell, to get the maximum emotional response from the reader. And finally, they increase the dynamics of the text. Some grammar rules may be skipped in descriptive paragraphs, but only for the sake of imagery.
Expository: It is not an easy task to write an expository paragraph, especially if you are an amateur in the subject. These paragraphs explain how something works or what the reader is to do to make it work. Such paragraphs demand a certain knowledge. Nevertheless, writing them is a great exercise to understand the material, because you keep learning when you teach."""
sentences = sent_tokenize(text)
sentence = sentences[0]
s0, s1, s2 = sentences[:3]

for sent in sentences:
    kw = keywordextract(sent, classifier_token, tokenizer)
    print(sent)
    if len(kw) > 0:
        print(">>",kw)



########################
"""
python keyword-extractor.py --path "model.pt" --sentence "The IOB format (short for inside, outside, beginning) is a common tagging format for tagging tokens in a chunking task in computational linguistics (ex. named-entity recognition)."

python keyword-extractor.py --path "model.pt" --sentence "Some students space paragraphs"

python keyword-extractor.py --path "model.pt" --sentence "trying to separate points when the process of writing is over."

python keyword-extractor.py --path "model.pt" --sentence "Such paragraphs demand a certain knowledge."
"""