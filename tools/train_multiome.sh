#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

prefix=$1

nohup python -u train_multiome.py \
    --submission_save_path=/home/jxf/code/kaggle_MSCI/results/${prefix}_partial_submission_multiome.pkl \
    --ckpt_save_path=/home/jxf/code/kaggle_MSCI/checkpoints/${prefix}_multi \
    --log_dir=/home/jxf/code/kaggle_MSCI/logs/${prefix}_multi \
    --batch_size=4096 \
    --input_dim=512 \
    --n_components=512 \
    --lr=1e-3 \
    --epoch=300 \
    --hidden_dim=1024 \
    --output_dim=23418 \
    > ./logs/${prefix}_multi.log 2>&1 &


