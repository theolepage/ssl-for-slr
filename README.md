# ssl-for-slr

Collection of **self-supervised** models for **speaker and language recognition** tasks.

## Models

- **CPC**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)
-  **LIM/GIM**: [Learning Speaker Representations with Mutual Information](https://arxiv.org/pdf/1812.00271.pdf)
-  **SimCLR**: [Contrastive Self-Supervised Learning for Text-Independent Speaker Verification](https://sci-hub.mksa.top/10.1109/icassp39728.2021.9413351)
-  **MoCo**: [Self-supervised Text-independent Speaker Verification using Prototypical Momentum Contrastive Learning](https://arxiv.org/pdf/2012.07178.pdf)

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

*Note: `trials` file of VoxCeleb1 can be download [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt) and `wav.scp` files were generated with Kaldi `make_voxceleb1.pl`/`make_voxceleb2.pl` scripts.*

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

Start self-supervised training with `python train.py configs/cpc-base.json`.

Then, you can evaluate model on speaker verification (EER, minDCF) with `python evaluate.py configs/cpc-base.json`.

## To-Do

- [ ] fix AudioAugmentation/cache_features
- [ ] is online eval faster? (40s) => remove tqdm
- [ ] Find NaN values

---

- [ ] Reproduce results of SimCLR
    - Possible differences with original SimCLR implem:
        - [ ] Data augmentation is currently disabled
        - [ ] Thin-ResNet34 implem (SE layer)
        - [ ] Loss implem
        - [ ] Channel invariant loss is currently disabled

- [ ] Experiment with different architectures and VICReg
- [ ] Cite articles in README
- [ ] Remote utt2spk and provide scripts to generate wav.scp files

---

- [ ] Use dataclass and YAML for model configs
- [ ] CPC/LIM: @tf.function warning when doing tensor[1, :]
- [ ] Fix warning loading weights not used
- [ ] Create custom training loop (https://stackoverflow.com/questions/57971007/tensorflow-2-0-display-progress-bar-in-custom-training-loop)
- [ ] Allow restore optimizer
