# SSL-for-SLR

Framework to train a speech encoder in a **self-supervised** way for **speaker and language recognition** tasks.

## To-Do

- [ ] Rewrite logs and push, start trainings on GPU
- [ ] evaluate.py
    - Learning curves
    - Speaker ID: scores, confusion matrix, error analysis, t-SNE
    - Speaker verification on VoxCelebs
    - Same test set?

---

- [ ] Improve encoder (SincConv) and classifier (dropout, normalization)
- [ ] Improve config: encoder, multiple training type (pretext, downstream)
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