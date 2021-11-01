# ssl-for-slr

Collection of **self-supervised** models for **speaker and language recognition** tasks.

## Models

- **[`sslforslr.models.cpc.CPC`](https://github.com/theolepage/ssl-for-slr/blob/master/sslforslr/models/cpc/CPC.py)**  
  "Representation Learning with Contrastive Predictive Coding" ([arxiv](https://arxiv.org/pdf/1807.03748.pdf))  
  *Aaron van den Oord, Yazhe Li, Oriol Vinyals*

- **[`sslforslr.models.lim.LIM`](https://github.com/theolepage/ssl-for-slr/blob/master/sslforslr/models/lim/LIM.py)**  
  "Learning Speaker Representations with Mutual Information" ([arxiv](https://arxiv.org/pdf/1812.00271.pdf))  
  *Mirco Ravanelli, Yoshua Bengio*

- **[`sslforslr.models.simclr.SimCLR`](https://github.com/theolepage/ssl-for-slr/blob/master/sslforslr/models/simclr/SimCLR.py)**  
  "Contrastive Self-Supervised Learning for Text-Independent Speaker Verification" ([sci-hub](https://sci-hub.mksa.top/10.1109/icassp39728.2021.9413351))  
  *Haoran Zhang, Yuexian Zou, Helin Wang*

- **[`sslforslr.models.moco.MoCo`](https://github.com/theolepage/ssl-for-slr/blob/master/sslforslr/models/moco/MoCo.py)**  
  "Self-supervised Text-independent Speaker Verification using Prototypical Momentum Contrastive Learning" ([arxiv](https://arxiv.org/pdf/2012.07178.pdf))  
  *Wei Xia, Chunlei Zhang, Chao Weng, Meng Yu, Dong Yu*

## Datasets

[VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) are used for our experiments. A folder `data` must be created at the root of the project with the structure shown below.

```
data
├── voxceleb1_test
│   ├── trials
│   └── wav.scp
├── voxceleb1_train
│   └── wav.scp
└── voxceleb2_train
    └── wav.scp
```

*Note: `trials` file of VoxCeleb1 can be download [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt) and `wav.scp` files were generated using [Kaldi scripts](https://github.com/kaldi-asr/kaldi/tree/master/egs/sitw/v1/local) (make_voxceleb1.pl and make_voxceleb2.pl).*

The format of `wav.scp` files must follow the one detaild below.

```
id00012-21Uxsk56VDQ-00001 /path/to/VoxCeleb2/dev/aac/id00012/21Uxsk56VDQ/00001.wav
id00012-21Uxsk56VDQ-00002 /path/to/VoxCeleb2/dev/aac/id00012/21Uxsk56VDQ/00002.wav
...
id09272-u7VNkYraCw0-00026 /path/to/VoxCeleb2/dev/aac/id09272/u7VNkYraCw0/00026.wav
id09272-u7VNkYraCw0-00027 /path/to/VoxCeleb2/dev/aac/id09272/u7VNkYraCw0/00027.wav
```

Regarding data augmentation, we rely on [MUSAN](http://www.openslr.org/17/) and [Room Impulse Response and Noise Database](https://www.openslr.org/28/).

## Usage

Start self-supervised training with `python train.py configs/cpc-base.yml`.

Then, you can evaluate model on speaker verification (EER, minDCF) with `python evaluate.py configs/cpc-base.yml`.

## To-Do

- [ ] Experiment with different architectures (VICReg)
    - [ ] loss = nce + 0.1 * vic
    - [ ] loss = vic
    - [ ] Ablation study on VICReg hyper-params
- [ ] Train other models (MoCo/XVectorEncoder, CPC/CPCEncoder, LIM/SincEncoder, Wav2Spk)
- [ ] Pytorch implementation of best model
- [ ] Provide script to download and prepare data

---

- [ ] CPC/LIM: @tf.function warning when doing tensor[1, :]
- [ ] Fix warning when loading: some weights are not used
- [ ] Allow restore optimizer
- [ ] Pytorch implementation
