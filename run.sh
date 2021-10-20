model_config_path=$1

export CUDA_VISIBLE_DEVICES=0,1
python train.py $model_config_path

python evaluate.py $model_config_path