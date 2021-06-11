# Trainings

1. Train all self-supervised models
    - CPC
    - PASE
    - wave2vec 2.0
    - CPC-VQ
2. Evaluate all on speaker recognition
3. Keep only best model
    - Evaluate speaker and language recognition
    - With supervised and random baseline
4. Ablation study on best model (change architecture, vary hyperparameters, layer norm, data augment, bidirectional)
5. Data-efficient evaluation
    - train only classifier
    - compare classifier trained on MFCC and SSL features
    - make sure equivalent # of params for both models
    - different training data ratio (1, 2, 5, 10, 20, 50, 100)
    - create table and graph like "Data-Efficient Image Recognition with Contrastive Predictive Coding"