"""
Code from https://github.com/huggingface/pytorch-transformers/blob/master/examples/lm_finetuning/pregenerate_training_data.py
"""
import collections
import json
import shelve
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from random import choice, randint, random, randrange, shuffle
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm, trange

from transformers import BertTokenizer
from text_seg import create_training_json

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    #parser.add_argument("--epochs_to_generate", type=int, default=1,
    #                    help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    #parser.add_argument("--short_seq_prob", type=float, default=0.1,
    #                    help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    
    args.output_dir.mkdir(exist_ok=True)

    if args.num_workers > 1:
        writer_workers = Pool(min(args.num_workers))
        arguments = [(args)]
        writer_workers.starmap(create_training_json, arguments)
    else:
        create_training_json(args, tokenizer)


if __name__ == '__main__':
    main()
