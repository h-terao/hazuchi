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
- chex
- rich

Most of dependncies are automatically installed when install Hazuchi.
However, you must install jax before the Hazuchi installation.

```bash
pip install git+https://github.com/h-terao/hazuchi
```

## Examples

```python
from hazuchi import Trainer, callbacks

trainer = Trainer(
    out_dir="./results",
    max_epochs=100,
    train_fn,
    val_fn,
    test_fn,
    callbacks=callbacks,
    prefetch=True
)

train_state = trainer.fit(...)
```