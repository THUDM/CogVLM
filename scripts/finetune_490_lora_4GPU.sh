#! /bin/bash
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

NUM_GPUS_PER_WORKER=4
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="cogvlm-base-490"
VERSION="base"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 1288 \
    --lora_rank 10 \
    --use_lora \
    --local_tokenizer lmsys/vicuna-7b-v1.5 \
    --version $VERSION"

OPTIONS_SAT="SAT_HOME=~/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

train_data="./archive_split/train"
valid_data="./archive_split/valid"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 800 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${valid_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 200 \
       --eval-interval 200 \
       --save "./checkpoints" \
       --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config scripts/test_config_bf16_490.json \
       --skip-init \
       --seed 2023
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_demo.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x