#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

voxceleb1_trials=data/voxceleb1_test/trials # downloaded by make_voxceleb1.pl
voxceleb1_root=/work2/home/ing2/datasets/VoxCeleb1
voxceleb2_root=/work2/home/ing2/datasets/VoxCeleb2
musan_root=/work2/home/ing2/datasets/musan

model_config_path=$1
expname=test_training
stage=6

# Stage 0: Prepare train and test data directories
# Train+CV = VoxCeleb2 dev set
# Test     = VoxCeleb1 test set
if [ $stage -le 0 ]; then
    echo "=== Stage 0: Prepare train and test data directories ==="
    
    log=exp/make_voxceleb
    
    #$train_cmd $log/make_voxceleb2_dev.log local/make_voxceleb2.pl $voxceleb2_root dev data/train
    $train_cmd $log/make_voxceleb1.log local/make_voxceleb1.pl $voxceleb1_root data

    echo -e "\n"
fi

# Stage 1: Make MFCCs and compute VAD decisions
if [ $stage -le 1 ]; then
    echo "=== Stage 1: Make MFCCs and compute VAD decisions ==="
    
    for name in train voxceleb1_test; do
        local/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf \
            --nj 40 --cmd "$train_cmd" data/${name} exp/make_mfcc data/mfcc
        local/fix_data_dir.sh data/${name}
 
        local/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
            data/${name} exp/make_vad data/mfcc
        local/fix_data_dir.sh data/${name}
    done
  
    echo -e "\n"
fi

# Stage 2: Augment audio training data (reverberation, noise, music, and babble).
if [ $stage -le 2 ]; then
    echo "=== Stage 2: Augment audio training data ==="
  
    log=exp/augmentation
    frame_shift=0.01
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

    if [ ! -d "RIRS_NOISES" ]; then
        # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
        wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
        unzip rirs_noises.zip
    fi

    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

    # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
    # additive noise here.
    $train_cmd $log/reverberate_data_dir.log python steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        data/train data/train_reverb
    cp data/train/vad.scp data/train_reverb/
    local/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
    rm -rf data/train_reverb
    mv data/train_reverb.new data/train_reverb

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    $train_cmd $log/make_musan.log local/make_musan.sh $musan_root data

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
        utils/data/get_utt2dur.sh data/musan_${name}
        mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done

    # Augment with musan_noise
    $train_cmd $log/augment_musan_noise.log python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
    # Augment with musan_music
    $train_cmd $log/augment_musan_music.log python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
    # Augment with musan_speech
    $train_cmd $log/augment_musan_speech.log python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

    # Combine reverb, noise, music, and babble into one directory.
    local/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
  
    echo -e "\n"
fi

# Stage 3: Merge training data and augmented training data
if [ $stage -le 3 ]; then
    echo "=== Stage 3: Merge training data and augmented training data ==="
  
    # Take a random subset of the augmentations
    local/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
    local/fix_data_dir.sh data/train_aug_1m

    # Make MFCCs for the augmented data.  Note that we do not compute a new
    # vad.scp file here.  Instead, we use the vad.scp from the clean version of
    # the list.
    local/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
        data/train_aug_1m exp/make_mfcc data/mfcc

    # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
    # double the size of the original clean list.
    local/combine_data.sh data/train_combined data/train_aug_1m data/train
  
    echo -e "\n"
fi

# Stage 4: Generate features from data (applies CMVN and remove nonspeech frames)
if [ $stage -le 4 ]; then
    echo "=== Stage 4: Generate features from data ==="
  
    # Note that this is somewhat wasteful, as it roughly doubles the amount
    # of training data on disk. After creating training examples, this can be removed.
  
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" --compress false \
        data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
    local/fix_data_dir.sh data/train_combined_no_sil

    local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" --compress false \
        data/train data/train_no_sil exp/train_no_sil
    local/fix_data_dir.sh data/train_no_sil

    local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" --compress false \
        data/voxceleb1_test data/voxceleb1_test_no_sil exp/voxceleb1_test_no_sil
    local/fix_data_dir.sh data/voxceleb1_test_no_sil

    echo -e "\n"
fi

# Stage 5: Filter samples and create val set
if [ $stage -le 5 ]; then
    echo "=== Stage 5: Filter samples and create val set ==="
  
    log=exp/processed
    mkdir -p $log
    
    # Remove utterances with less than 400 frames
    awk 'NR==FNR{a[$1]=$2;next}{if(a[$1]>=400)print}' data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > $log/utt2spk 
  
    # Create spk2num_frames (useful for balancing training)
    awk '{if(!($2 in a))a[$2]=0;a[$2]+=1;}END{for(i in a)print i,a[i]}' $log/utt2spk > $log/spk2num 
  
    # Create train (90%) and cv (10%) utterances list
    awk -v seed=$RANDOM 'BEGIN{srand(seed);}NR==FNR{a[$1]=$2;next}{if(a[$2]<10)print $1>>"exp/processed/train.list";else{if(rand()<=0.1)print $1>>"exp/processed/cv.list";else print $1>>"exp/processed/train.list"}}' $log/spk2num $log/utt2spk 

    # Split frontend feats into feats_front_train.scp and feats_front_val.scp
    awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $log/train.list data/train_combined_no_sil/feats.scp | shuf > $log/feats_front_train.scp
    awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $log/cv.list data/train_combined_no_sil/feats.scp | shuf > $log/feats_front_val.scp

    # Merge feats for backend
    cat data/train_no_sil/feats.scp > $log/feats_back.scp
    cat data/voxceleb1_test_no_sil/feats.scp >> $log/feats_back.scp

    # Map speakers to labels (utt2spkid)
    awk 'BEGIN{s=0;}{if(!($2 in a)){a[$2]=s;s+=1;}print $1,a[$2]}' $log/utt2spk > $log/utt2spkid

    echo -e "\n"
fi

# Stage 6-8: Train feature extractor neural network
if [ $stage -le 6 ]; then  
    echo "=== Stage 6-8: Train feature extractor neural network ==="

    expdir=exp/training/$expname/
    mkdir -p $expdir

    #$train_cmd $expdir/train.log
    python ../train.py $model_config_path

    echo -e "\n"
fi

train_utt2spk=data/train/utt2spk
train_spk2utt=data/train/spk2utt
test_utt2spk=data/voxceleb1_test/utt2spk

backend_log=exp/backend/$expname/
mkdir -p $backend_log

# Stage 9: Extract speaker embeddings
if [ $stage -le 9 ]; then
    echo "=== Stage 9: Extract speaker embeddings ==="
  
    expdir=exp/decode/$expname/
    mkdir -p $expdir
    
    #$train_cmd $expdir/decode.log
    python ../extract_embeddings.py \
        exp/processed/feats_back.scp \
        $expdir/embeddings.ark \
	$model_config_path
    
    awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $test_utt2spk $expdir/embeddings.ark > $backend_log/test.iv
    awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $train_utt2spk $expdir/embeddings.ark > $backend_log/train.iv

    echo -e "\n"
fi

# Stage 10: Train backend PLDA model
if [ $stage -le 10 ]; then
    echo "=== Stage 10: Train backend PLDA model ==="
    
    # Compute the mean vector for centering the evaluation ivectors.
	$train_cmd $backend_log/compute_mean.log \
		ivector-mean ark:$backend_log/train.iv \
		$backend_log/mean.vec || exit 1;

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    lda_dim=200
    $train_cmd $backend_log/lda.log \
        ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
        "ark:ivector-subtract-global-mean ark:$backend_log/train.iv ark:- |" \
        ark:$train_utt2spk $backend_log/transform.mat || exit 1;

    # Train the PLDA model.
    $train_cmd $backend_log/plda.log \
        ivector-compute-plda ark:$train_spk2utt \
        "ark:ivector-subtract-global-mean ark:$backend_log/train.iv ark:- | transform-vec $backend_log/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
        $backend_log/plda || exit 1;
  
    echo -e "\n"
fi

# Stage 11: Score model on test set
if [ $stage -le 11 ]; then
    echo "=== Stage 11: Score model on test set ==="
    
    $train_cmd $backend_log/voxceleb1_test_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $backend_log/plda - |" \
        "ark:ivector-subtract-global-mean $backend_log/mean.vec ark:$backend_log/test.iv ark:- | transform-vec $backend_log/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $backend_log/mean.vec ark:$backend_log/test.iv ark:- | transform-vec $backend_log/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $backend_log/scores_voxceleb1_test || exit 1;

    echo -e "\n"
fi

# Stage 12: Show evaluation metrics (EER, minDCF)
if [ $stage -le 12 ]; then
    echo "=== Stage 12: Show evaluation metrics ==="
    
    eer=`compute-eer <(python local/prepare_for_eer.py $voxceleb1_trials $backend_log/scores_voxceleb1_test) 2> /dev/null`
    mindcf1=`python local/compute_min_dcf.py --p-target 0.01 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
    echo "EER: $eer%"
    echo "minDCF(p-target=0.01): $mindcf1"
    echo "minDCF(p-target=0.001): $mindcf2"
fi
