# SSL-for-SLR

Framework to train a speech encoder in a **self-supervised** way for **speaker and language recognition** tasks.

## Training

1. Self-supervised: encoder + CPC/LIM/GIM
2. Supervised (transfer learning, fine-tuning): encoder (pre-trained) + classifier
3. Supervised: encoder (scratch) + classifier

## Evaluation

1. Learning curves
2. Speaker ID
    - Scores
    - Confusion matrix
    - Error analysis
    - t-SNE
3. Speaker verification on VoxCelebs

## To-Do

- [ ] train_spkid.py (steps 2 and 3) args: path to config. If no model in config => supervised
- [ ] Push, start trainings on GPU
- [ ] evaluate.py (same test set?)

---

- [ ] Merge self-supervised modules and add modules (GIM, LPS, FBANKS)
- [ ] Implement new ideas (more params in encoder, bi-directional) and benchmark
- [ ] Data augmentation / preprocessing step

---

- [ ] CPC: negative samples from same speaker + current and other sentences, accuracy only on last timestep?
- [ ] Comment code
- [ ] Properly set seed

## References

- [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)
- [Learning Problem-agnostic Speech Representationsfrom Multiple Self-supervised Tasks](https://arxiv.org/pdf/1904.03416.pdf)
- [Multi-task self-supervised learning for Robust Speech Recognition](https://arxiv.org/pdf/2001.09239.pdf)
- [Learning Speaker Representations with Mutual Information](https://arxiv.org/pdf/1812.00271.pdf)
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)