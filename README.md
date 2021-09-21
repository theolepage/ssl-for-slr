# SSL-for-SLR

Framework to train a speech encoder in a **self-supervised** way for **speaker and language recognition** tasks.

The aim is to train a speech encoder by using multiple self-supervised modules as shown on figure below.

![model_multitask](https://raw.githubusercontent.com/theolepage/ssl-for-slr/master/docs/model_multitask.png)

## Features

- Configurable speech encoders (1D conv layers, GRU, skip connections, [SincNet](https://arxiv.org/abs/1808.00158))
- Self-supervised models:
    - [Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf) *(unidirectional or bidirectional)*
    - [vq-wav2vec](https://arxiv.org/pdf/1910.05453.pdf)
    - [Wav2Vec 2.0](https://arxiv.org/pdf/2006.11477.pdf)
    - [Local Info Max (LIM)](https://arxiv.org/pdf/1812.00271.pdf) and Global Info Max (GIM)
    - [PASE](https://arxiv.org/pdf/1904.03416.pdf) and [PASE+](https://arxiv.org/pdf/2001.09239.pdf) with the following workers: *Waveform*, *LPS*, *MFCC*, *CPC*, *LIM* and *GIM*
- Evaluation on speaker recognition, speaker verification, language recognition and data-efficiency
- Handle *LibriSpeech* and *VoxLingua107* datasets
- Speech augmentation module *(reverberation, noise, frequency and temporal masks, clipping, ...)*
- Modular configuration files

## Usage

### Install dependencies (inside a virtual env)

1. `virtualenv ~/ssl-for-slr-env && source ~/ssl-for-slr-env/bin/activate`
2. `pip install -r requirements.txt`

*Type `deactivate` to exit the virtual env after use.*

### Train model on pretext task

```
python train.py configs/cpc-v1.json
```

*Multiple config files are located in the `config/` folder.*

### Evaluate model on downstream task *(speaker or language recognition)*

1. Train a classifier on top of the previsouly trained encoder: `python train_evaluate.py configs/cpc-v1.json`.
2. Use notebook `evaluate.ipnyb` to evaluate metrics obtained on the downstream task.

## To-Do

- [ ] Merge with branch tmp-package
- [ ] Config GRU dim CPC, multi GPU training, tensorboard
- [ ] Link with Kaldi
- [ ] Change name?
- [ ] Improve dataset: do not store audio cache in checkpoints/model/
- [ ] Delete speaker id evaluation? choose in train_evaluate type of classifier and random, surpervised baseline
- [ ] Fix error end training saving history.npy

---

- [ ] Use dataclass and YAML for all configs
- [ ] Create custom training loop (https://stackoverflow.com/questions/57971007/tensorflow-2-0-display-progress-bar-in-custom-training-loop)
- [ ] Fix warning loading weights not used
- [ ] CPC/LIM: @tf.function warning when doing tensor[1, :]
- [ ] Allow restore optimizer