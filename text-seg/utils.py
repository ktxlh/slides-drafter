import os
import json
import torch
from nltk.tokenize import sent_tokenize

def remove_non_printable(s):
        s = s.encode('ascii', errors='ignore').decode('ascii')
        s = '\n'.join([ss for ss in s.split('\n') if len(ss) > 40])
        return s

def traverse_json_dir(json_dir, return_docs):
    rtn = []
    num_sentences_counter = 0
    print("*** Traversing ***")
    for root, dirs, files in os.walk(json_dir):
        print("# of json files in total:",len(files))
        files.sort()
        for fname in files:
            obj = json.load(open(os.path.join(json_dir, fname)))
            sections = []
            for secs in obj['now']['sections']:
                text = remove_non_printable(secs['text'])
                if len (text) > 40:
                    sentences = sent_tokenize(text)
                    num_sentences_counter += len(sentences)
                    sections.append(sentences)
            if return_docs:
                rtn.append(sections)
            else:
                rtn.extend(sections)
    print("# of sentences in total:",num_sentences_counter)
    print("--- Traversing done ---")
    return rtn

def keywordextract(sentence, model):
    """
    Usage:
        keywordextract(args.sentence, model_dir)
    
    from: https://github.com/ibatra/BERT-Keyword-Extractor/blob/master/keyword-extractor.py
    """
    text = sentence
    tkns = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model.eval()
    prediction = []
    logit = model(tokens_tensor, token_type_ids=None,
                                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, axis=2)])
    for k, j in enumerate(prediction[0]):
        if j==1 or j==0:
            print(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k], j)
            