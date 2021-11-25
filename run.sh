#!/bin/env sh

model_config_path=$1

mkdir data
python prepare_data.py data

export CUDA_VISIBLE_DEVICES=0,1
python train.py $model_config_path

python evaluate.py $model_config_path 2> /dev/null