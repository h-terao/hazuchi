from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import chex


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
