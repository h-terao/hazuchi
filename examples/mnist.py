"""MNIST example."""
from __future__ import annotations
import functools
import math
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, core
from flax.training.train_state import TrainState
import optax
import einops
import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.append(".")
import hazuchi

tf.config.experimental.set_visible_devices([], "GPU")


def get_dataset(batch_size: int = 32, split: str = "train"):
    def map_fn(item):
        x = einops.repeat(item["image"], "N H W C -> N H W (repeat C)", repeat=3)
        # if split == "train":
        # x = tf.image.random_flip_left_right(x)
        return {"images": tf.cast(x, tf.float32) / 255.0, "labels": item["label"]}

    data = tfds.load("mnist", split=split, shuffle_files=True)
    N = 60000 if split == "train" else 10000
    if split == "train":
        iter_len = N // batch_size
    else:
        iter_len = math.ceil(N / batch_size)

    if split == "train":
        data = data.cache().repeat().shuffle(N)
    data = data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(map_fn, tf.data.AUTOTUNE)
    if split != "train":
        data = data.cache().repeat()
    data = data.prefetch(tf.data.AUTOTUNE)
    data_iter = data.as_numpy_iterator()
    return data_iter, iter_len


class CNN(linen.Module):
    """A simple CNN model.

    Modify from https://github.com/google/flax/blob/main/examples/mnist/train.py
    """

    @linen.compact
    def __call__(self, x, train):
        x = linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = linen.relu(x)
        x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = linen.relu(x)
        x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = linen.Dense(features=256)(x)
        x = linen.relu(x)
        x = linen.Dense(features=10)(x)
        return x


class TrainState(TrainState):
    model_state: core.FrozenDict = None


def init_fn(rng, batch):
    images = batch["images"]
    model_state, params = CNN().init({"params": rng}, images, train=True).pop("params")
    tx = optax.adabelief(0.001)
    return TrainState.create(
        apply_fn=CNN().apply,
        params=params,
        tx=tx,
        model_state=model_state,
    )


def train_fn(train_state: TrainState, batch):
    images, labels = batch["images"], batch["labels"]
    labels = jax.nn.one_hot(labels, num_classes=10)

    @functools.partial(jax.grad, has_aux=True)
    def grad_fn(params):
        variables = {"params": params, **train_state.model_state}
        logits, new_model_state = train_state.apply_fn(
            variables, images, train=True, mutable=["batch_stats"]
        )

        loss = optax.softmax_cross_entropy(logits, labels).mean()
        accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)).mean()

        return loss, (new_model_state, {"loss": loss, "accuracy": accuracy})

    grads, (new_model_state, scalars) = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grads, model_state=new_model_state)
    return new_train_state, scalars


def eval_fn(train_state, batch):
    images, labels = batch["images"], batch["labels"]
    labels = jax.nn.one_hot(labels, num_classes=10)

    variables = {"params": train_state.params, **train_state.model_state}
    logits = train_state.apply_fn(variables, images, train=False)

    loss = optax.softmax_cross_entropy(logits, labels).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)).mean()
    return {"loss": loss, "accuracy": accuracy}


def main():
    print("Create trainer.")
    trainer = hazuchi.Trainer(
        "out",
        max_epochs=10,
        train_fn=train_fn,
        val_fn=eval_fn,
        callbacks={
            "best_acc": hazuchi.callbacks.BestValue(
                monitor="val/accuracy",
                mode="max",
            ),
            "logger": hazuchi.callbacks.JsonLogger(),
            "printer": hazuchi.callbacks.PrintMetrics(
                [
                    "epoch",
                    "train/loss",
                    "train/accuracy",
                    "val/loss",
                    "val/accuracy",
                    "val/accuracy_best",
                    "elapsed_time",
                ]
            ),
            "pbar": hazuchi.callbacks.ProgressBar(),
            "timer": hazuchi.callbacks.Timer(),
            "early_stopping": hazuchi.callbacks.EarlyStopping(
                monitor="train/accuracy",
                mode="max",
            ),
            "best_snap": hazuchi.callbacks.Snapshot(
                "best.ckpt",
                monitor="val/accuracy",
                mode="max",
                load_before_testing=True,
            ),
            "last_snap": hazuchi.callbacks.Snapshot(
                "last.ckpt",
                load_before_fitting=True,
            ),
        },
        prefetch=True,  # If you use CPU or TPU, set prefetch=False
    )

    # Logging hyperparams
    trainer.log_hyperparams({"dummy_params": 1})

    rng = jr.PRNGKey(9)

    print("Data loader.")
    train_loader, train_len = get_dataset(128, "train")
    test_loader, test_len = get_dataset(512, "test")

    print("Initialize")
    rng, init_rng = jr.split(rng)
    batch = next(iter(train_loader))
    train_state = init_fn(init_rng, batch)

    # Start training.
    print("Train")
    train_state = trainer.fit(
        train_state,
        train_loader,
        test_loader,
        train_len=train_len,
        val_len=test_len,
    )

    # Testing
    print("Test")
    trainer.test(train_state, test_loader, test_len)


if __name__ == "__main__":
    main()
