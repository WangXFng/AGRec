# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/

python  LLaMA-1B-train.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 4 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json\
    --wandb_run_name test\
    --temperature 1.0