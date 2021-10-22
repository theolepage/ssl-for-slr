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

- [ ] Reproduce results of SimCLR
    - Possible differences with original SimCLR implem:
        - [ ] x = self.torchfb(x)+1e-6
        - [ ] x = x.log()
        - [ ] x = self.instancenorm(x).unsqueeze(1)
        - [ ] Thin-ResNet34 implem
        - [ ] loss implem
        - [ ] data augmentation is currently disabled
        - [ ] channel invariant loss is currently disabled

- [ ] Experiment with different architectures and VICReg
- [ ] Explain data preparation / reproduction + cite articles in README

---

- [ ] Use dataclass and YAML for model configs
- [ ] CPC/LIM: @tf.function warning when doing tensor[1, :]
- [ ] Fix warning loading weights not used
- [ ] Create custom training loop (https://stackoverflow.com/questions/57971007/tensorflow-2-0-display-progress-bar-in-custom-training-loop)
- [ ] Allow restore optimizer