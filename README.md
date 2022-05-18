<div align="center">

# Hazuchi

An all-in-one training tool for JAX/Flax for my projects. <br>
"Hazuchi" comes from the Japanese deity 天羽槌雄神, who is worshiped as the deity of cloth and weaving. <br>
This project is now in progress.
</div>

<br><br>

## Introduction

Hazuchi is a Python library that provides a simple trainer and callbacks for JAX/Flax.


## Dependencies

- Python >= 3.7
- jax >= 0.3.4
- flax
- chex
- rich

Most of dependncies are automatically installed when install Hazuchi.
However, you must install jax before the Hazuchi installation.

```bash
pip install git+https://github.com/h-terao/hazuchi
```

## Modules
Hazuchi consist of the following modules.

- Trainer: A trainer class.
- Observation: A class for metric summarization.
- callbacks: Classes to extend the trainer.
- functional: Popular functions.
- torch_utils: PyTorch utilities to use data.DataLoader with JAX.
- utils: Utilities.
- image
    - transforms: Functions to transform images. The pixel value should be in [0, 1].
