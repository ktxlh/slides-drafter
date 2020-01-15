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
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from utils import traverse_json_dir

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
# TODO Add argparse
num_train_epochs = 10
batch_size = 16 # TODO batch_size was 64 in the example
max_sent_len = 40

weight_decay = 0.0 ###
learning_rate = 5e-5 ###
adam_epsilon = 1e-8 ###
warmup_steps = 0 ###
max_grad_norm = 1.0 ###

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
tag = '{}-{}-{}-{}-{}'.format(*json_dir.replace('-','_').split('/')[-2:], num_train_epochs, batch_size, max_sent_len)
model_dir = "/home/shanglinghsu/ml-camp/models/"+tag
loss_dir = "/home/shanglinghsu/ml-camp/losses/"+tag
for d in [model_dir, loss_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# mini-json: Subset with only 7822*.json and 7823*.json
tokenizer_encode_plus_parameters = {
    'max_length' : max_sent_len,
    'pad_to_max_length' : 'right',
    'return_tensors' : 'pt',
}
seed = 666
def set_seed(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
PAD_ID = tokenizer.pad_token_id
"""
## TODO From pre_trained
model = model_class.from_pretrained('./directory/to/save/')  # re-load
tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load
"""

# Data (subset) -> Dataset
docs = traverse_json_dir(json_dir, return_docs=True)
inputs, labels = [],[]

# docs之間無關
for sections in docs:
    # secs之間是1
    for i in range(len(sections)-1):
        inputs.append(tokenizer.encode_plus(
            sections[i][-1], sections[i+1][0], **tokenizer_encode_plus_parameters
        )['input_ids'])
        labels.append(1)

    # sec內sents間是0
    for sents in sections:
        for i in range(len(sents)-1):
            inputs.append(tokenizer.encode_plus(
                sents[i], sents[i+1], **tokenizer_encode_plus_parameters
            )['input_ids'])
            labels.append(0)

data = TensorDataset(torch.cat(inputs), torch.tensor(labels))
n_test = int(len(labels)*0.2)
n_train = len(labels) - n_test
train_set, valid_set = random_split(data, [n_train, n_test])
train_generator = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
valid_generator = DataLoader(valid_set, sampler=RandomSampler(valid_set), batch_size=batch_size)

def test_model(model, device, generator,tokenizer): # generator
    # TODO change run.sh seq_len
    
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

test_model(model, device, valid_generator,tokenizer)
