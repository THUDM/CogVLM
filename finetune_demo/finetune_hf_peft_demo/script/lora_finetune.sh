#!/bin/bash

OMP_NUM_THREADS=96  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4  ../train_peft_lora.py \
  --dataset_dir "/share/home/zyx/Dataset/new_dataset" \
  --train_rate 0.8 \
  --lr 2e-4 \
  --num_epochs 20 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --model_dir "/share/home/zyx/Models/cogagent-chat-hf" \
  --tokenizer_dir "/share/official_pretrains/hf_home/vicuna-7b-v1.5" \
  --save_dir "output" \
  --save_freq 5 \
  --lora_rank 64 \
  --lora_dropout 0.1 \
  --lora_alpha 32 \
  --max_length 350 \
