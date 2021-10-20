# ssl-for-slr

Collection of **self-supervised** models for **speaker and language recognition** tasks.

## Models

- **CPC**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)
-  **LIM/GIM**: [Learning Speaker Representations with Mutual Information](https://arxiv.org/pdf/1812.00271.pdf)
-  **SimCLR**: [Contrastive Self-Supervised Learning for Text-Independent Speaker Verification](https://sci-hub.mksa.top/10.1109/icassp39728.2021.9413351)
-  **MoCo**: [Self-supervised Text-independent Speaker Verification using Prototypical Momentum Contrastive Learning](https://arxiv.org/pdf/2012.07178.pdf)

## Usage

Start self-supervised training with `python train.py configs/cpc-base.json`.

Then, you can evaluate model on speaker verification (EER, minDCF) with `python evaluate.py configs/cpc-base.json`.

## To-Do

- [ ] Refactor project
    - [ ] Data: check similar (padding) [30min]
    - [ ] Evaluate: check works [30min]
    - [ ] Model: clamp W, init -5 10, check similar encoder, mfcc [1h]
    - [ ] Start SimCLR training [30min]

- [ ] Reproduce results of SimCLR
    - [ ] If not working => use voxceleb_trainer implem
    - [ ] Add data augmentation
    - [ ] Evaluate: add minDCF
- [ ] Experiment with VICReg

---

- [ ] Explain data preparation / reproduction + cite articles in README
- [ ] Use dataclass and YAML for model configs
- [ ] CPC/LIM: @tf.function warning when doing tensor[1, :]
- [ ] Fix warning loading weights not used
- [ ] Create custom training loop (https://stackoverflow.com/questions/57971007/tensorflow-2-0-display-progress-bar-in-custom-training-loop)
- [ ] Allow restore optimizer