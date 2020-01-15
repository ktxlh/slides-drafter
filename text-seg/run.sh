# Dependencies:
#   Scikit-learn (pip install -U --user scikit-learn)
JSON=/home/shanglinghsu/ml-camp/wiki-vandalism/json/
PREGENERATE=/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen
MAX_SEQLEN=50

python pregen.py --json $JSON --bert_model bert-base-cased --output_dir $PREGENERATE/ --max_seq_len $MAX_SEQLEN

python finetune.py --pregenerated_data $PREGENERATE/ --bert_model bert-base-cased --output_dir $PREGENERATE/models/ --fp16 --epochs 3

# Instructions fixing run-out-of-memory error when finetuning on single GPU
# https://github.com/huggingface/pytorch-transformers/tree/master/examples/lm_finetuning