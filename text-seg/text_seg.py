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

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
# TODO Add argparse
num_train_epochs = 10
batch_size = 16 # TODO batch_size was 64 in the example
max_sent_len = 40
limit_paragraphs = 10 ### TODO

weight_decay = 0.0 ###
learning_rate = 5e-5 ###
adam_epsilon = 1e-8 ###
warmup_steps = 0 ###
max_grad_norm = 1.0 ###

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
tag = '{}-{}-{}-{}-{}-{}'.format(*json_dir.replace('-','_').split('/')[-2:], num_train_epochs, batch_size, max_sent_len, limit_paragraphs)
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
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
PAD_ID = tokenizer.pad_token_id
"""
## From pre_trained
model = model_class.from_pretrained('./directory/to/save/')  # re-load
tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load
"""

# Data (subset) -> Dataset
sections = traverse_json_dir(json_dir, toke_to_sent=True, limit_paragraphs=limit_paragraphs)

sent_secs = []
for i in range(len(sections)):
    sent_secs.extend(zip([i]*len(sections[i]), sections[i]))

# TODO: Too slow. 1)switch to index combinations 2)Materialize to list 3)random.choice 
combs = combinations(sent_secs, 2)
inputs, labels = [],[]
for (l1,s1), (l2,s2) in tqdm(combs): # TODO Tqdm -> Trange
    inputs.append(tokenizer.encode_plus(
        s1, s2, **tokenizer_encode_plus_parameters
    )['input_ids'])
    labels.append(int(l1 == l2))

data = TensorDataset(torch.cat(inputs), torch.tensor(labels))
n_test = int(len(labels)*0.2)
n_train = len(labels) - n_test
train_set, valid_set = random_split(data, [n_train, n_test])
train_generator = DataLoader(train_set, sampler=RandomSampler(data), batch_size=batch_size)
valid_generator = DataLoader(valid_set, sampler=RandomSampler(data), batch_size=batch_size)

# Fine-tune model

## Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_generator)
)

print("***** Running training *****")
tr_loss, vl_loss = [],[]
global_step = 0
epochs_trained = 0
#steps_trained_in_current_epoch = 0
model.zero_grad()
train_iterator = trange(
    epochs_trained, int(num_train_epochs), desc="Epoch"
)
set_seed(seed)  # Added here for reproducibility
for _ in train_iterator:
    epoch_iterator = tqdm(train_generator, desc="Iteration")
    for step, local_batch, local_labels in enumerate(epoch_iterator): # TODO What's the enumerate() for?

        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        model.train()
        outputs = model(local_batch, labels=local_labels)
        loss = outputs[0]
        loss.backward()
        tr_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) ###
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in valid_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            outputs = model(local_batch, labels=local_labels)
            loss = outputs[0]
            vl_loss.append(loss)

print("*** losses ***")
for lt,lv in zip(tr_loss, vl_loss):
    print('{:5f}\t{:5f}'.format(lt,lv))

with open(loss_dir+'loss.txt','w') as f:
    f.write('\n'.join(['{}\t{}' for lt,lv in zip(tr_loss, vl_loss)])+'\n')

# Save model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

"""
Footnotes
<1> Weird tables example (# of sentences: 28)
TEXT     Frank Grillo as Leo Barnes
Elizabeth Mitchell as Senator Charlie Roan
Christy Coco as Young Charlie Roan
Mykelti Williamson as Joe Dixon
Joseph Julian Soria as Marcos Dali
Betty Gabriel as Laney Rucker
Terry Serpico as Earl Danzinger
Raymond J. Barry as Caleb Warrens
Edwin Hodge as Dante Bishop
Cindy Robinson as Purge Announcement Voice
Kyle Secor as Minister Edwidge Owens
Liza Coln-Zayas as Dawn
David Aaron Baker as NFFA Press Secretary Tommy Roseland
Christopher James Baker as Harmon James
Brittany Mirabil as Kimmy
Juani Feliz as Kimmy's friend
Roman Blat as Lead Russian Murder Tourist purger in Uncle Sam costume (credited as "Uncle Sam")
Jamal Peters as Crips leader (credited as "Gang Leader with Dying Friend")
J. Jewels as Political Debater
Matt Walton as Reporter #1

"""
