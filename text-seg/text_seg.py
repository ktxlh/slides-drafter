"""
Goal: Segment text into sections
Task: Classify a pair of sentences into "from the same section or not"

Data dir (shanglinghsu@): ~/ml-camp/wiki-vandalism/json
"""
import argparse
import json
import os
import random
import re
import string
from collections import Counter
from itertools import combinations
from summa.keywords import keywords as get_keywords
from summa.summarizer import summarize

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize  # , texttiling
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

stop_words = set(stopwords.words('english')) 

# mini-json: Subset with only 7822*.json and 7823*.json
json_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json" # Should be json
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/" #TODO Unused. Comment it out

#for d in [model_dir]: #loss_dir
#    if not os.path.exists(d):
#        os.makedirs(d)

seed = 666
def set_seed(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(seed)
    
def get_inputs_labels(sections):
    # One doc consists of several sections
    # sections = [["I am sent 1.","Sent 2."],["3rd sent.","FOURTH S."]]
    inputs, labels = [],[]

    # secs之間是1
    for i in range(len(sections)-1):
        inputs.append((sections[i][-1], sections[i+1][0]))
        labels.append(1)

    # sec內sents間是0
    # negative sampling
    # 只要不是該section最後一句，都平均地可能被選到
    population = [(sec_num, sent_num) for sec_num, sents in enumerate(sections) for sent_num in range(len(sents)-1)]
    choices = random.choices(
        population, k = min(len(sections)-1, len(population)))
    for (sec_num, sent_num) in choices:
        inputs.append((sections[sec_num][sent_num], sections[sec_num][sent_num+1]))
        labels.append(0)
    return inputs, labels

# pregen.py
def create_instances_from_json(max_seq_length, tokenizer, json_dir):
    instances = []

    tokenizer_encode_plus_parameters = { 'max_length' : max_seq_length, 'pad_to_max_length' : 'right', 'add_special_tokens' : True, }

    docs = traverse_json_dir(json_dir, return_docs=True)

    print("*** Batch encoding ***")
    for sections in docs:
        inputs, labels = get_inputs_labels(sections)
    
        for start in range(0, len(inputs), 16):
            end = min(len(inputs), start+16)

            inputs_sub, labels_sub = inputs[start:end], labels[start:end]
            enc =  tokenizer.batch_encode_plus(inputs_sub, **tokenizer_encode_plus_parameters)
        
            for input_id, token_type_id, label in zip(enc['input_ids'], enc['token_type_ids'], labels_sub):
                instances.append({
                    "tokens": tokenizer.convert_ids_to_tokens(input_id),
                    "segment_ids": token_type_id,
                    "is_random_next": label,
                    "masked_lm_positions": [],
                    "masked_lm_labels": []
                })

    print("--- Batch encoding done ---")

    random.shuffle(instances)
    return instances

def create_training_json(args, tokenizer):
    epoch_filename = args.output_dir / "epoch_0.json"
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        doc_instances = create_instances_from_json(
            args.max_seq_len, tokenizer, args.json_dir)
        doc_instances = [json.dumps(instance) for instance in doc_instances]
        for instance in doc_instances:
            epoch_file.write(instance + '\n')
            num_instances += 1
    metrics_file = args.output_dir / "epoch_0_metrics.json"
    with metrics_file.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))

class TextSplitter():
    """
    Please make sure that the dependencies 
    in the begginning of this doc are 
    properly installed before using this
    """
    def __init__(self, model_dir, tok_model_path):
        self.tokenizer = traTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)#, output_attentions=True,'bert-base-cased')
        
        self.token_classifier = torch.load(tok_model_path)
        self.ppb_tokenizer = pytTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        #self.texttiling = texttiling.TextTilingTokenizer()

    def split(self, text):
        """
        The main API of this code.
        Splits text according to:
        1) '\n'
        2) Semantic

        : text: str -- Normal text input
        : title: str -- The key sentence of the whole text or "Title"
        : segments: list(str) -- Each str is a semantic segment.
        : keywords: list(list(str)) -- Each list corresponds 
            to a segment, containing 0 or more keywords
        : subtitles: list(str) -- Each str may be "Title" or one key sentence.
        """
        title = summarize(text).split('\n')[0]
        if len(title) == 0:
            title = "Title"

        segments = []
        keywords = []
        subtitles = []
        
        ### 1) paragraph (split by '\n')
        paragraphs = [t for t in text.split('\n') if len(t) > 0]
        for paragraph in paragraphs:
            
            #self.texttiling.tokenize(paragraph)

            sents = sent_tokenize(paragraph)
            if len(sents) == 0:
                continue

            segments.append([])
            keywords.append([])
            for i in range(len(sents)-1):
                # "Current" and "next" sentences
                input_ids = self.tokenizer.encode_plus(sents[i],sents[i+1], return_tensors='pt')['input_ids']
                logits = self.model(input_ids)[0]
                input_ids = input_ids.squeeze(0)
                logits = logits.squeeze(0)
                
                ## Update list with this result
                segments[-1].append(sents[i])
                keywords[-1].extend(self.extract_keywords(sents[i]))
                
                ## Split paragraph
                ### 2) semantic segment
                softmax = torch.nn.functional.softmax(logits, dim=0)
                argmax = softmax.argmax().item()
                if argmax: # 1 if diff; 0 otherwise
                    segments.append([])
                    keywords.append([])
            
            # The last sentence
            segments[-1].append(sents[-1])
            keywords[-1].extend(self.extract_keywords(sents[-1]))

        segments = [' '.join(sents) for sents in segments]
        subtitles = [summarize(segment).split('\n')[0] for segment in segments]
        return title, segments, keywords, subtitles
    
    def extract_keywords(self, sentence):
        keywords = []
        keywords.extend(self._extract_keywords_helper(sentence))
        for subsent in sentence.split(','):
            tmp = self._extract_keywords_helper(subsent)
            keywords.extend(tmp)

        # Incorporate info from tool
        keywords.extend(get_keywords(sentence).split('\n'))
        
        # Remove duplicates
        keywords = list(set(keywords))
        for w1 in keywords.copy():
            for w2 in keywords:
                if w2 != w1 and w2 in w1:
                    keywords.remove(w2)
                    break
        for w1 in keywords.copy():
            for w2 in keywords:
                if w2 != w1 and w2 in w1:
                    keywords.remove(w2)
                    break

        # Only 0 ~ 4 keywords for now (for formatting)
        keywords = [w for w in keywords if not w in stop_words and not w in string.punctuation]
        occurence_count = Counter(keywords)
        keywords = [w for (w,_) in occurence_count.most_common(4)]
        return keywords
        
    def _extract_keywords_helper(self, sub_sentence):
        text = sub_sentence
        words = word_tokenize(text) # ["I","sleep","."]
        tkns, tkn_words = [],[]
        for word in words:
            subwords = self.ppb_tokenizer.tokenize(word)
            tkn_words.extend([
                word for j in range(len(subwords))
            ])
            tkns.extend(subwords)  # ["sle","##ep"]
        
        indexed_tokens = self.ppb_tokenizer.convert_tokens_to_ids(tkns)
        segments_ids = [0] * len(tkns)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
        self.token_classifier.eval()
        self.token_classifier.to(self.device)
        prediction = []
        logit = self.token_classifier(tokens_tensor, token_type_ids=None,
                                    attention_mask=segments_tensors)
        logit = logit.detach().cpu().numpy()
        prediction.extend([list(p) for p in np.argmax(logit, axis=2)])
        keywords = []
        for k, j in enumerate(prediction[0]):
            # tag2idx = {'B': 0, 'I': 1, 'O': 2}
            if j==1 or j==0:
                keywords.append(tkn_words[k].lower())
                #print(tkn_words[k], j)
        return keywords

"""
def test_model(model, device, tokenizer): # generator
    
    inputs, labels = get_inputs_labels(json_dir)
    data = TensorDataset(torch.cat(inputs), torch.tensor(labels))
    n_test = int(len(labels)*0.2)
    n_train = len(labels) - n_test
    train_set, valid_set = random_split(data, [n_train, n_test])
    train_generator = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
    valid_generator = DataLoader(valid_set, sampler=RandomSampler(valid_set), batch_size=batch_size)
    
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

# Load model
model = BertForSequenceClassification.from_pretrained(model_dir)#'bert-base-cased')
tokenizer = traTokenizer.from_pretrained(model_dir)#'bert-base-cased')

#test_model(model, device, tokenizer)
"""
