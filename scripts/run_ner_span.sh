CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cner"

if [ $# = "0" ];then
    echo "command: run_ner_span.sh train/test!"
    exit 0;
fi

if [ $1 = "train" ]; then
    echo "-------------------- train -----------------------"
    python3 run_ner_span.py \
      --model_type=bert \
      --model_name_or_path=$BERT_BASE_DIR \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_adv \
      --do_lower_case \
      --loss_type=ce \
      --data_dir=$DATA_DIR/${TASK_NAME}/ \
      --train_max_seq_length=128 \
      --eval_max_seq_length=512 \
      --per_gpu_train_batch_size=24 \
      --per_gpu_eval_batch_size=24 \
      --learning_rate=2e-5 \
      --num_train_epochs=20.0 \
      --logging_steps=-1 \
      --save_steps=-1 \
      --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
else
    echo "-------------------- test -----------------------"
    python3 run_ner_span.py \
      --model_type=bert \
      --model_name_or_path=$BERT_BASE_DIR \
      --task_name=$TASK_NAME \
      --data_dir=$DATA_DIR/${TASK_NAME}/ \
      --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
      --do_predict
fi
