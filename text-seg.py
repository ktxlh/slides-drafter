"""
Goal: Segment text into paragraphs
Task: Classify a pair of sentences into "from the same paragraph or not"

Data dir (shanglinghsu@): ~/ml-camp/wiki-vandalism/json
"""
import os
import json
import torch
from transformers import BertForSequenceClassification

JSON_DIR = "/home/shanglinghsu/ml-camp/wiki-vandalism/json/."
NUM_PAR_TRAIN = 1000

# Load model
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
"""
## From pre_trained
model = model_class.from_pretrained('./directory/to/save/')  # re-load
tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load
"""

# Data (subset) -> Dataset
## traverse root directory, and list directories as dirs and files as files
paragraphs = []
for root, dirs, files in os.walk(JSON_DIR):
    print("# of json files in total:",len(files))
    for fname in files:
        json_obj = json.load(open(fname))
        print(json_obj)
        break
        #paragraphs.append()
        #if len(paragraphs) == NUM_PAR_TRAIN:
        #    break
print("# of paragraphs loaded:", len(paragraphs))

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
# Test model 1
# Fine-tune model
# Test model 2

# Save model
"""
model.save_pretrained('./directory/to/save/')  # save
tokenizer.save_pretrained('./directory/to/save/')  # save
"""