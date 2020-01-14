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
import json
import os
import random
import re
import string
from itertools import combinations

from nltk.tokenize import sent_tokenize

import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import BertForSequenceClassification, BertTokenizer

# Parameters
max_epochs = 10
params = {'batch_size': 16, # TODO batch_size was 64 in the example
          'shuffle': True,
          'num_workers': 6}
random.seed(666)
torch.manual_seed(666)
json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
# mini-json: Subset with only 7822*.json and 7823*.json
num_par_train = 10

SENT_TEMP = "[CLS] {} [SEP] {}"


# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
"""
## From pre_trained
model = model_class.from_pretrained('./directory/to/save/')  # re-load
tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load
"""

def remove_non_printable(s):
    return s.encode('ascii', errors='ignore').decode('ascii')

# Data (subset) -> Dataset
## traverse root directory, and list directories as dirs and files as files
sections = []
for root, dirs, files in os.walk(json_dir):
    print("# of json files in total:",len(files))
    files.sort()
    for fname in files:
        obj = json.load(open(os.path.join(json_dir, fname)))
        for secs in obj['now']['sections']:
            text = remove_non_printable(secs['text'])
            if len (text) > 0:
                sentences = sent_tokenize(text)
                if len(sentences) > 10:
                    continue # Some tables are weird <1>
                print(len(sentences))
                sections.append(sentences)
        if len(sections) >= num_par_train: ### TODO Use more data later
            break

print("# of sections loaded:", len(sections))

## Sentences from same section
same_sec_pairs = []
for section in sections:
    for comb in combinations(section, 2):
        if random.random() < 0.5: ### TODO How many from same/diff sections?
            same_sec_pairs.append(tokenizer.encode(SENT_TEMP.format(comb[0], comb[1])))
### TODO 改！這個要先processed再做embedding
same_sec_input_ids = torch.tensor(same_sec_pairs)
same_sec_labels = torch.ones([same_sec_input_ids.size()[0],1]) # same number of 1s

print(same_sec_pairs[0])
print(same_sec_input_ids[0])
print(same_sec_labels[0])

## Sentences from diff sections
diff_sec_pairs = []
for comb in combinations(sections, 2):
    if random.random < 0.5:
        sent1 = random.choice(comb[0])
        sent2 = random.choice(comb[1])
        diff_sec_pairs.append(tokenizer.encode(SENT_TEMP.format(sent1, sent2))))
diff_sec_input_ids = torch.tensor(diff_sec_pairs)
diff_sec_labels = torch.zeros([diff_sec_input_ids.size()[0],1]) # same number of 0s

print(diff_sec_pairs[0])
print(diff_sec_input_ids[0])
print(diff_sec_labels[0])

inputs_ids = torch.cat((same_sec_input_ids, diff_sec_input_ids), 0)
labels = torch.cat((same_sec_labels, diff_sec_labels), 0)
ds = Dataset(inputs_ids, labels)
train_set, test_set = torch.utils.data.random_split(ds, [int(num_par_train*0.8), int(num_par_train*0.2)])


# Fine-tune model
for 


# Test fine-tuned model

"""
input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
labels = None

# If you used to have this line in pytorch-pretrained-bert:
loss = model(input_ids, labels=labels)

# Now just use this line in transformers to extract the loss from the output tuple:
outputs = model(input_ids, labels=labels)
loss = outputs[0]

# In transformers you can also have access to the logits:
loss, logits = outputs[:2]

# And even the attention weights if you configure the model to output them (and other outputs too, see the docstrings and documentation)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(input_ids, labels=labels)
loss, logits, attentions = outputs
"""


# Save model
"""
model.save_pretrained('./directory/to/save/')  # save
tokenizer.save_pretrained('./directory/to/save/')  # save
"""

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
