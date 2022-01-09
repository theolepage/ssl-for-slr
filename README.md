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

[VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) are used for our experiments and we rely on [MUSAN](http://www.openslr.org/17/) and [Room Impulse Response and Noise Database](https://www.openslr.org/28/) for data augmentation.

To download, extract and prepare all datasets run `python prepare_data.py data/`.  The `data/` directory will have the structure detailed below.

```
data
├── musan_split/
├── simulated_rirs/
├── voxceleb1/
├── voxceleb2/
├── trials
├── voxceleb1_train_list
└── voxceleb2_train_list
```

Trials and train lists files are also automatically created with the following formats.

- `trials`
    ```
    1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
    ...
    0 id10309/0cYFdtyWVds/00005.wav id10296/Y-qKARMSO7k/00001.wav
    ```

- `voxceleb1_train_list` and `voxceleb2_train_list`
    ```
    id00012 voxceleb2/id00012/21Uxsk56VDQ/00001.wav
    ...
    id09272 voxceleb2/id09272/u7VNkYraCw0/00027.wav
    ```

*Please refer to `prepare_data.py` script if you want further details about data preparation.*

## Usage

Start self-supervised training with `python train.py configs/cpc-base.yml`.

Then, you can evaluate model on speaker verification (EER, minDCF) with `python evaluate.py configs/cpc-base.yml`.

## To-Do

- [ ] Pytorch implementation
- [ ] Change repo/project name -> `ssl-for-sv`?
- [ ] Make sure other models work (MoCo/XVectorEncoder, CPC/CPCEncoder, LIM/SincEncoder, Wav2Spk)
- [ ] Get model name with config filename
- [ ] CPC/LIM: @tf.function warning when doing tensor[1, :]
- [ ] Fix warning when loading: some weights are not used
- [ ] Allow restore optimizer