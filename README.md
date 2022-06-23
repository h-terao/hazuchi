<div align="center">

# Hazuchi

Hazuchi is a high-level API for JAX/Flax. <br>
"Hazuchi" comes from the Japanese deity 天羽槌雄神, who is worshiped as the deity of cloth and weaving. <br>
This project is now in progress.
</div>

<br><br>


## Dependencies

- Python >= 3.7
- jax >= 0.3.4
- chex
- rich
- tqdm

Most of dependncies are automatically installed when install Hazuchi.
However, you must install jax before the Hazuchi installation.

```bash
pip install git+https://github.com/h-terao/hazuchi
```

## Examples

```python
from hazuchi import Trainer, callbacks

def train_fn(train_state, batch):
    ...
    scalars = {"loss": ...}
    return new_train_state, scalars

def val_fn(train_state, batch):
    ...
    return scalars

# Define trainer
trainer = Trainer(
    out_dir="./results",
    max_epochs=100,
    train_fn,
    val_fn,
    callbacks=callbacks,
)

# Start training.
train_state = trainer.fit(train_state, train_iter, val_iter)

# Start testing
trainer.test(train_state, test_iter, test_len)
```