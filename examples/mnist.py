from __future__ import annotations
import functools
import math

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen, training, core
import optax
import einops
import hub
import tensorflow as tf

import hazuchi

tf.config.experimental.set_visible_devices([], "GPU")


def get_dataset(path: str, batch_size: int = 32, train: bool = False):
    def map_fn(item):
        x = einops.repeat(item["image"], "N H W -> N H W C", C=3)
        if train:
            x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], mode="REFLECT")
            x = tf.image.random_crop(x, size=28)
            x = tf.image.random_flip_left_right(x)
        return {"images": tf.cast(x, tf.float32) / 255.0, "labels": item["label"]}

    data: tf.data.Dataset = hub.load(path)
    if train:
        iter_len = len(data) // batch_size
    else:
        iter_len = math.ceil(len(data) / batch_size)

    data = data.tensorflow()
    data = data.cache()
    if train:
        data = data.repeat()
    data = data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(map_fn, tf.data.AUTOTUNE)
    if not train:
        data = data.repeat()
    data_iter = data.as_numpy_iterator()
    return data_iter, iter_len


class CNN(linen.Module):
    @linen.compact
    def __call__(self, x, train):
        return linen.Sequential(
            [
                linen.Conv(32, [3, 3], use_bias=False),
                linen.BatchNorm(use_running_average=not train),
                jax.nn.relu(),
                linen.Conv(64, [3, 3], use_bias=False),
                linen.BatchNorm(use_running_average=not train),
                jax.nn.relu(),
                linen.Conv(128, [3, 3], use_bias=False),
                linen.BatchNorm(use_running_average=not train),
                jax.nn.relu(),
                linen.Conv(256, [3, 3], use_bias=False),
                linen.BatchNorm(use_running_average=not train),
                jax.nn.relu(),
                lambda x: jnp.mean(x, axis=(1, 2)),
                linen.Dense(128),
                jax.nn.relu(),
                linen.Dense(10),
            ]
        )(x)


class TrainState(training.train_state.TrainState):
    model_state: core.FrozenDict


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
    new_train_state = train_state.apply_gradients(grads, model_state=new_model_state)
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

    trainer = hazuchi.Trainer(
        "out",
        max_epochs=100,
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
            "snapshot": hazuchi.callbacks.Snapshot(
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
    )

    rng = jr.PRNGKey(9)

    train_loader, train_len = get_dataset("hub://activeloop/mnist-train", 128, train=True)
    test_loader, test_len = get_dataset("hub://activeloop/mnist-train", 512)

    rng, init_rng = jr.split(rng)
    train_state = init_fn(init_rng, next(train_loader))

    # Start training.
    train_state = trainer.fit(
        train_state,
        train_loader,
        test_loader,
        train_len=train_len,
        val_len=test_len,
    )

    # Testing
    summary = trainer.test(train_state, test_loader, test_len)
    print(summary)


if __name__ == "__main__":
    main()
