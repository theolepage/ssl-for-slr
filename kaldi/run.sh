#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

voxceleb1_trials=data/voxceleb1_test/trials # created by make_voxceleb1.pl
voxceleb1_root=/diskssd1/ing2/datasets/VoxCeleb1
voxceleb2_root=/diskssd1/ing2/datasets/VoxCeleb2

model_config_path=$1
expname=test_training
stage=3

# Stage 0: Prepare train and test data directories
# Train+CV = VoxCeleb2 dev set
# Test     = VoxCeleb1 test set
if [ $stage -le 0 ]; then
    echo "=== Stage 0: Prepare train and test data directories ==="
    
    log=exp/make_voxceleb
    
    $train_cmd $log/make_voxceleb2_dev.log local/make_voxceleb2.pl $voxceleb2_root dev data/train
    $train_cmd $log/make_voxceleb1.log local/make_voxceleb1.pl $voxceleb1_root data

    echo -e "\n"
fi

# Stage 1: Generate features from data
if [ $stage -le 1 ]; then
    echo "=== Stage 1: Generate features from data ==="
  
    log=exp/encode
    
    #$train_cmd $log/encode.log
    python ../cache_features.py data/train
    #$train_cmd $log/encode.log
    python ../cache_features.py data/voxceleb1_test
    
    echo -e "\n"
fi

# Stage 2: Train feature extractor neural network
if [ $stage -le 2 ]; then  
    echo "=== Stage 2: Train feature extractor neural network ==="

    #expdir=exp/training/$expname/
    #mkdir -p $expdir

    export CUDA_VISIBLE_DEVICES=0,1

    #$train_cmd $expdir/train.log
    python ../train.py $model_config_path

    echo -e "\n"
fi

backend_log=exp/backend/$expname/
mkdir -p $backend_log

# Stage 3: Extract speaker embeddings
if [ $stage -le 3 ]; then
    echo "=== Stage 3: Extract speaker embeddings ==="
  
    #expdir=exp/decode/$expname/
    #mkdir -p $expdir
    
    export CUDA_VISIBLE_DEVICES=0
    
    #$train_cmd $expdir/decode.log
    python ../extract_embeddings.py \
        data/voxceleb1_test/feats.scp \
        $backend_log/test.ark \
	$model_config_path

    echo -e "\n"
fi

# Stage 4: Score model on test set
if [ $stage -le 4 ]; then
    echo "=== Stage 4: Score model on test set ==="
    
    python ../kaldi_evaluate.py \
        $voxceleb1_trials \
	$backend_log/test.scp \
	$backend_log/scores_voxceleb1_test

    echo -e "\n"
fi

# Stage 5: Show evaluation metrics (EER, minDCF)
if [ $stage -le 5 ]; then
    echo "=== Stage 5: Show evaluation metrics ==="
    
    eer=`compute-eer <(python local/prepare_for_eer.py $voxceleb1_trials $backend_log/scores_voxceleb1_test) 2> /dev/null`
    mindcf1=`python local/compute_min_dcf.py --p-target 0.01 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`

    echo "EER: $eer%"
    echo "minDCF(p-target=0.01): $mindcf1"
    echo "minDCF(p-target=0.001): $mindcf2"
fi
