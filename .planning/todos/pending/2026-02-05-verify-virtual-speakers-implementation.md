---
created: 2026-02-05T23:07
title: Verify virtual speakers implementation correctness
area: modules
files:
  - src/datamodules/components/common.py
  - configs/datamodule/multi_sv.yaml
  - src/modules/sv.py
---

## Problem

The latest commit (b604c24) introduced virtual speakers and mutually exclusive noise/reverb augmentation. Training logs show suspicious behavior that suggests something may be broken:

- Validation accuracy near zero and **decreasing** over epochs (0.0037 -> 0.0012)
- Validation loss stays very high (~14-17) and isn't improving
- Train loss drops (11.3 -> 0.97) suggesting overfitting or label mismatch
- Epoch 4 shows sudden train loss spike (0.97 -> 2.88) which could indicate instability

The train/valid divergence pattern (train loss dropping while valid accuracy decreases) strongly suggests a label mismatch issue — virtual speaker class IDs may not align between train and validation, or the number of output classes in the classifier head may not match the actual number of (real + virtual) speakers.

Key concerns to investigate:
1. Are virtual speaker class_ids consistent between train/valid splits?
2. Does the classifier head output dimension match total speakers (real + virtual)?
3. Is the augmentation pipeline corrupting features or labels?
4. Are virtual speaker samples correctly constructed (correct audio segments, proper labels)?

## Solution

TBD — needs investigation of the virtual speaker logic in the datamodule and how class IDs flow through to the loss function.
