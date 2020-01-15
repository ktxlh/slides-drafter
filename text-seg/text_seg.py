"""
Goal: Segment text into sections
Task: Classify a pair of sentences into "from the same section or not"

Data dir (shanglinghsu@): ~/ml-camp/wiki-vandalism/json


*** Before using this code ***
* default pytorch conda ve provided: conda activate pytorch
* pip install --user transformers nltk
* In python, run this
  >>> import nltk
  >>> nltk.download('punkt')

"""
import argparse
import json
import os
import random
import re
import string
from itertools import combinations

from tqdm import tqdm, trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset,
                              random_split)
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)
from utils import traverse_json_dir

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
tag = '{}-{}-{}-{}'.format(*json_dir.replace('-','_').split('/')[-2:], batch_size, max_seq_length)
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/"

for d in [model_dir]: #loss_dir
    if not os.path.exists(d):
        os.makedirs(d)

# mini-json: Subset with only 7822*.json and 7823*.json
tokenizer_encode_plus_parameters = {
    'max_length' : max_seq_length,
    'pad_to_max_length' : 'right',
    'return_tensors' : 'pt',
}
seed = 666
def set_seed(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)

# Load model
model = BertForSequenceClassification.from_pretrained(model_dir)#'bert-base-cased')
tokenizer = BertTokenizer.from_pretrained(model_dir)#'bert-base-cased')

def get_inputs_labels(json_dir):
    # Data (subset) -> Dataset
    docs = traverse_json_dir(json_dir, return_docs=True)
    inputs, labels = [],[]
    # docs之間無關
    for sections in docs:
        # secs之間是1
        for i in range(len(sections)-1):
            inputs.append((sections[i][-1], sections[i+1][0]))
            labels.append(1)

        # sec內sents間是0
        for sents in sections:
            for i in range(len(sents)-1):
                inputs.append((sents[i], sents[i+1]))
                labels.append(0)
    return inputs, labels

# pregen.py
def create_instances_from_json(
        max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    instances = []
    
    inputs, labels = get_inputs_labels(json_dir)
    encodings =  tokenizer.batch_encode_plus(inputs, **tokenizer_encode_plus_parameters)

    for inp, lab in all_in:
        
        instance = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "is_random_next": is_random_next,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}
        instances.append(instance)


def test_model(model, device, tokenizer): # generator
    # TODO change run.sh seq_len
    
    """
    inputs, labels = get_inputs_labels(json_dir)
    data = TensorDataset(torch.cat(inputs), torch.tensor(labels))
    n_test = int(len(labels)*0.2)
    n_train = len(labels) - n_test
    train_set, valid_set = random_split(data, [n_train, n_test])
    train_generator = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
    valid_generator = DataLoader(valid_set, sampler=RandomSampler(valid_set), batch_size=batch_size)
    """
    # Transfer to GPU
    vl_loss = []
    model = model.to(device)

    for local_batch, local_labels in valid_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        outputs = model(local_batch, labels=local_labels)
        loss, logits = outputs[:2]
        vl_loss.append(loss)
        print("loss:",loss)
        for b,l,ll in zip(local_batch, logits, local_labels):
            print(l.tolist(),'\t',ll.tolist(), '\t', tokenizer.decode(b))
        break

#test_model(model, device, tokenizer)
