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
from utils import traverse_json_dir, remove_non_printable
# TODO Check imports (are copy-pasted for now)

json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json

