#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

prefix=$1

nohup python -u train_citeseq.py \
    --submission_save_path=/home/jxf/code/kaggle_MSCI/results/${prefix}_cite_submission_citeseq.pkl \
    --ckpt_save_path=/home/jxf/code/kaggle_MSCI/checkpoints/${prefix}_cite \
    --log_dir=/home/jxf/code/kaggle_MSCI/logs/${prefix}_cite \
    --batch_size=256 \
    --input_dim=240 \
    --n_components=240 \
    --lr=5e-4 \
    --epoch=100 \
    --hidden_dim=128 \
    --output_dim=140 \
    > ./logs/${prefix}_cite.log 2>&1 &


