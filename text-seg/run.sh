export TRAIN_FILE=/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/train.raw
export TEST_FILE=/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/test.raw

python run_lm_finetuning.py \
    --output_dir=output \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --model_type=bert \ 
    --model_name_or_path=bert_base_cased
    # roberta?