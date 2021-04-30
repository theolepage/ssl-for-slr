# SSL-for-SLR

Framework to train a speech encoder in a **self-supervised** way for **speaker and language recognition** tasks.

## Usage

### Install dependencies (inside a virtual env)

1. `virtualenv ~/ssl-for-slr-env && source ~/ssl-for-slr-env/bin/activate`
2. `pip install -r requirements.txt`

*Type `deactivate` to exit the virtual env after use.*

### Start self-supervised training

Multiple config files are located in the `config/` folder.

```
python train.py configs/cpc-v1.json
```

### Evaluate the model

To evaluate the model we train a speaker id classifier on top of the pre-trained encoder with the following steps:

1. `python train_spkid.py configs/cpc-v1.json`
2. `python evaluate.py configs/cpc-v1.json`

## To-Do

- [ ] Improve encoder (SincConv) and classifier (dropout, normalization)
- [ ] Improve config: encoder, multiple training type (pretext, downstream)
- [ ] Merge self-supervised modules

---

- [ ] Implement new modules (GIM, LPS, FBANKS)
- [ ] Implement new ideas (more params, CPC sampling, bi-directional) and benchmark
- [ ] Data augmentation / preprocessing step
- [ ] Evaluate: speaker verification on VoxCelebs
- [ ] Evaluate: language recognition

---

- [ ] Add training time in history.npy
- [ ] Fix warning loading weights not used
- [ ] LibriSpeech: correct way of extracting a frame from an audio file?
- [ ] CPC/LIM: @tf.function warning
- [ ] Ability to resume training (load/save weights of optimizer) (https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state)
- [ ] Comment code
- [ ] Properly set seed

## References

- [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)
- [Learning Problem-agnostic Speech Representationsfrom Multiple Self-supervised Tasks](https://arxiv.org/pdf/1904.03416.pdf)
- [Multi-task self-supervised learning for Robust Speech Recognition](https://arxiv.org/pdf/2001.09239.pdf)
- [Learning Speaker Representations with Mutual Information](https://arxiv.org/pdf/1812.00271.pdf)
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)