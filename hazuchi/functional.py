from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import chex


def one_hot(
    labels: chex.Array,
    num_classes: int,
    label_smoothing: float = 0,
    dtype: chex.ArrayDType = jnp.float32,
) -> chex.Array:
    """Convert to soft labels.

    Args:
        labels: Label indices.
        num_classes (int): Number of classes.
        label_smoothing (float): Label smoothing value.
        dtype: JAX array dtype.
    """
    off_value = label_smoothing / num_classes
    on_value = 1.0 - label_smoothing + off_value
    labels = jax.nn.one_hot(labels, num_classes, dtype=dtype)
    labels = jnp.where(labels > 0.5, on_value, off_value)
    return labels


def accuracy(logits: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Top-k accuracy score.

    Args:
        logits: logits.
        labels: one-hot labels.

    Returns:
        Top-k accuracy.

    Example:
        >>> logits = jnp.array([[0.1, 0.9], [0.9, 0.1], [1.0, 0.0]])
        >>> labels = jnp.array([0, 0, 0])
        >>> accuracy(logits, labels).tolist()
        [0, 1, 0]
        >>> accuracy(logits, labels).mean()
        0.666
        >>> accuracy(logits, logits).mean()  # Use soft-target.
        1.0
    """
    assert logits.ndim in (labels.ndim, labels.ndim + 1)

    preds = jnp.argsort(logits)[..., -k:]
    if logits.ndim == labels.ndim:
        labels = jnp.argmax(labels, axis=-1, keepdims=True)
    else:
        labels = labels[..., None]
    return jnp.sum(preds == labels, axis=-1)


def absolute_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    """Computes the absolute difference between inputs and targets."""
    return jnp.abs(inputs - targets)


def squared_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    return jnp.square(inputs - targets)


def huber_loss(inputs: chex.Array, targets: chex.Array, delta: float) -> chex.Array:
    x = inputs - targets
    return jnp.where(
        jnp.abs(x) < delta,
        jnp.square(x) / 2,
        delta * (jnp.abs(x) - delta / 2),
    )


def cross_entropy(
    logits: chex.Array, labels: chex.Array, label_smoothing: float = 0.0
) -> chex.Array:
    """The cross-entropy loss."""
    assert logits.ndim in [labels.ndim, labels.ndim + 1]

    num_classes = logits.shape[-1]
    if logits.ndim == labels.ndim + 1:
        labels = one_hot(labels, num_classes, label_smoothing, dtype=logits.dtype)

    assert logits.shape == labels.shape
    log_preds = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_preds, axis=-1)


def kl_div(logits: chex.Array, targets: chex.Array, log_targets: bool = False) -> chex.Array:
    """The Kullback-Leibler divergence.

    Args:
        logits: Logit array.
        targets: Target array.
        log_targets (bool): If True, targets is considered as the log prob.
    """
    log_preds = jax.nn.log_softmax(logits)
    targets, log_targets = jax.lax.cond(
        log_targets, lambda t: (jnp.exp(t), t), lambda t: (t, jnp.log(t)), targets
    )

    return jnp.sum(targets * (log_preds - log_targets), axis=-1)


def js_div(logits: chex.Array, targets: chex.Array, log_targets: bool = False) -> chex.Array:
    """The Jensen-Shannon divergence.

    Args:
        logits: Logit array.
        targets: Target array.
        log_targets (bool): If True, targets is considered as the log prob.
    """
    assert logits.shape == targets.shape

    preds = jax.nn.softmax(logits)
    targets, log_targets = jax.lax.cond(
        log_targets, lambda t: (jnp.exp(t), t), lambda t: (t, jnp.log(t)), targets
    )
    y = (preds + targets) / 2
    return (kl_div(logits, y) + kl_div(log_targets, y)) / 2.0


def triplet_loss(
    inputs: chex.Array,
    positive_inputs: chex.Array,
    negative_inputs: chex.Array,
    loss_fun: Callable[[chex.Array, chex.Array], chex.Array],
    margin: float = 1.0,
) -> chex.Array:
    x = loss_fun(inputs, positive_inputs) - loss_fun(inputs, negative_inputs) + margin
    x = jax.nn.relu(x)
    return x


def weight_decay_loss(
    x: chex.PyTreeDef,
    penalty_fun: Callable[[chex.Array], chex.Array],
    filter_fun: Callable[[chex.Array], bool] | None = None,
) -> chex.Array:
    if filter_fun is None:
        filter_fun = lambda v: True  # noqa
    return sum([jnp.sum(penalty_fun(xi)) for xi in jax.tree_leaves(x) if filter_fun(xi)])


def charbonnier_penalty(x: chex.Array, eps: float = 0.001, alpha: float = 0.5) -> chex.Array:
    """The generalized Charbonnier penalty function.

    The generalized Charbonnier penalty function is defined as
        :math:`loss = (x^2 + e^2)^{1/alpha}`.

    Args:
        x: Input array.
    """
    return jnp.power(x**2 + eps**2, alpha)


def vat_noise(
    x: chex.Array,
    grad_fun: Callable[[chex.Array], chex.Array],
    axis: int | tuple[int, ...] | None = None,
    num_iters: int = 1,
    eps: float = 1.0,
) -> chex.Array:
    """Generates a noise for virtual adversarial training.

    Args:
        x (chex.Array): An initial noise.
        grad_fun (Callable): A function that compute gradients w.r.t x.
        axis (int, tuple of int, None): Axis to independently generate noises.
        num_iters (int): Number of backward computations to create noises.
        eps (float): A norm of noise.

    Returns:
        The adversarial noise.
    """

    @jax.jit
    def scan_fun(x: chex.Array, zero: chex.Array):
        new_x = grad_fun(x)
        new_x /= jnp.linalg.norm(new_x, ord=2, axis=axis, keepdims=True) + 1e-8
        return new_x, zero

    new_x, _ = jax.lax.scan(scan_fun, init=x, xs=jnp.zeros(num_iters))
    return new_x * eps


def permutate(x: chex.Array, indices: chex.Array, inv: bool = False) -> chex.Array:
    n = len(indices)
    if inv:
        indices = jnp.zeros_like(indices).at[indices].set(jnp.arange(n))
    return x[indices]
