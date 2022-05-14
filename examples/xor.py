"""Train a simple MLP by XOR data."""
import math
import functools

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import linen
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import optax
import chex

import hazuchi as hz
from hazuchi import functional as HF


def xor_data(rng: chex.PRNGKey, num_samples: int, batch_size: int):
    rng, new_rng = jrandom.split(rng)
    x = jrandom.uniform(new_rng, (2 * num_samples, 2), dtype="float32", minval=-1, maxval=1)
    t = jnp.logical_xor(x[:, 0] >= 0, x[:, 1] >= 0).astype("int")

    train_data = x[:num_samples], t[:num_samples]
    test_data = x[num_samples:], t[num_samples:]

    def loader(rng, data, train: bool = False):
        inputs, labels = data
        num_samples = len(inputs)

        num_main_samples = (num_samples // batch_size) * batch_size

        while True:
            if train:
                rng, new_rng = jrandom.split(rng)
                indices = jrandom.permutation(new_rng, num_samples)
                inputs = inputs[indices]
                labels = labels[indices]

            for batch in zip(
                inputs[:num_main_samples].reshape(-1, batch_size, 2),
                labels[:num_main_samples].reshape(-1, batch_size),
            ):
                yield batch

            if not train:
                yield inputs[num_main_samples:], labels[num_main_samples:]

    train_loader = loader(rng, train_data, True)
    test_loader = loader(rng, test_data, False)

    return train_loader, test_loader


class MLP(linen.Module):
    @linen.compact
    def __call__(self, x, train: bool = False):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x = linen.Dense(8, use_bias=False)(x)
        x = linen.BatchNorm(not train)(x)
        x = linen.relu(x)

        x = linen.Dense(2)(x)
        return x


class TrainState(TrainState):
    # If you want to use random function, register rng into train_state.
    rng: chex.PRNGKey = jrandom.PRNGKey(0)
    model_state: FrozenDict = FrozenDict()


@jax.jit
def train_fun(train_state: TrainState, batch):
    rng, new_rng = jrandom.split(train_state.rng)
    rng = hz.utils.bind_rng_to_host_device(rng, "batch", "device")

    inputs, labels = batch
    batch_size = len(inputs)

    @functools.partial(jax.value_and_grad, has_aux=True)
    def grad_fun(params):
        logits, new_model_state = train_state.apply_fn(
            {"params": params, **train_state.model_state},
            inputs,
            mutable=["batch_stats"],
            train=True,
        )

        loss = HF.cross_entropy(logits, labels).mean()
        accuracy = HF.accuracy(logits, labels).mean()

        observation = hz.Observation.create(
            {
                "loss": loss,
                "accuracy": accuracy,
            },
            batch_size,
        )

        return loss, (new_model_state, observation)

    aux, grads = grad_fun(train_state.params)
    new_model_state, observation = aux[1]

    grads = jax.lax.pmean(grads, axis_name="batch")
    new_model_state = jax.lax.pmean(new_model_state, axis_name="batch")

    new_train_state = train_state.apply_gradients(grads=grads, model_state=new_model_state, rng=new_rng)
    return new_train_state, observation


@jax.jit
def eval_fun(train_state: TrainState, batch):
    inputs, labels = batch
    batch_size = len(inputs)

    logits = train_state.apply_fn(
        {"params": train_state.params, **train_state.model_state},
        inputs,
        train=False,
    )

    return hz.Observation.create(
        {
            "loss": HF.cross_entropy(logits, labels).mean(),
            "accuracy": HF.accuracy(logits, labels).mean(),
        },
        batch_size,
    )


def main():
    rng = jrandom.PRNGKey(0)
    num_samples = 10000
    batch_size = 32

    rng, new_rng = jrandom.split(rng)
    train_data, test_data = xor_data(new_rng, num_samples, batch_size)

    rng, param_rng = jrandom.split(rng)
    model = MLP()
    model_state, params = model.init({"params": param_rng}, jnp.zeros((32, 2), dtype="float32")).pop("params")
    tx = optax.adafactor(learning_rate=0.001)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, model_state=model_state, rng=rng)

    trainer = hz.Trainer(
        train_fun=train_fun,
        eval_fun=eval_fun,
        max_epochs=100,
        callbacks={
            "print_metrics": hz.callbacks.PrintMetrics(
                ["epoch", "train/loss", "train/accuracy", "val/loss", "val/accuracy"]
            ),
            "json_logger": hz.callbacks.JsonLogger("out"),
            "progress_bar": hz.callbacks.ProgressBar(on_step=True, on_epoch=True),
        },
    )

    train_state = trainer.fit(
        train_state,
        train_data,
        test_data,
        train_steps_per_epoch=num_samples // batch_size,
        val_steps_per_epoch=math.ceil(num_samples / batch_size),
    )


if __name__ == "__main__":
    main()
