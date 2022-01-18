#!/bin/env sh

mkdir data
python prepare_data.py data

# export CUDA_VISIBLE_DEVICES=0,1

python train.py configs/vicreg_b256_1_1_0.yml
python train.py configs/vicreg_b256_1_1_0.1.yml
python train.py configs/vicreg_b256_1_1_0.04.yml
python train.py configs/vicreg_b256_1_0.5_0.1.yml