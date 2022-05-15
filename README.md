<div align="center">

# Hazuchi

An all-in-one training tool for JAX/Flax for my projects. <br>
"Hazuchi" comes from the Japanese deity 天羽槌雄神, who is worshiped as the deity of cloth and weaving.

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

### hazuchi
- Trainer: A trainer class.
- Observation: An observation to summarize metrics.
- callbacks: Callbacks to extend the trainer.
- functional: Functions.
- torch_utils: PyTorch utilities to use data loaders.
- image
    - transforms: <br> Image augmentation. The input images are expected as arrays that have a range of [0, 255]
