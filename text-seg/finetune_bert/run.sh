# Dependencies:
#   Scikit-learn (pip install -U --user scikit-learn)

CORPUS=/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/train.txt
PREGENERATE=/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen
MAX_SEQLEN=256

python pregen.py --train_corpus $CORPUS --bert_model bert-base-uncased --do_lower_case --output_dir $PREGENERATE/ --epochs_to_generate 3 --max_seq_len $MAX_SEQLEN

python finetune.py --pregenerated_data $PREGENERATE/ --bert_model bert-base-uncased --do_lower_case --output_dir $PREGENERATE/models/ --epochs 3 --train_batch_size 32 --gradient_accumulation_steps 4 | tee output.txt

# Instructions fixing run-out-of-memory error when finetuning on single GPU
# https://github.com/huggingface/pytorch-transformers/tree/master/examples/lm_finetuning