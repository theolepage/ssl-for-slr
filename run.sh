#!/bin/env sh

mkdir data
python prepare_data.py data

export CUDA_VISIBLE_DEVICES=0,1

python train.py configs/vicreg_b256.yml
python evaluate.py configs/vicreg_b256.yml

python train.py configs/vicreg_b512.yml
python evaluate.py configs/vicreg_b512.yml

python train.py configs/vicreg_b1024.yml
python evaluate.py configs/vicreg_b1024.yml

python train.py configs/vicreg_b2048.yml
python evaluate.py configs/vicreg_b2048.yml

python train.py configs/vicreg_b256_4xencoder.yml
python evaluate.py configs/vicreg_b256_4xencoder.yml

python train.py configs/vicreg_b256_2xmse.yml
python evaluate.py configs/vicreg_b256_2xmse.yml