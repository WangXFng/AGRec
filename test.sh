# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/ddp.json

# alphas = {
#     'Instruments': 0.4,
#     'Arts': 0.2,
#     'Games': 0.7,
#     'Yelp': 0.7,
# }

python  LLaMA-1B-test.py \
    --ckpt_path $OUTPUT_DIR \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.json\
    --alpha 0.4