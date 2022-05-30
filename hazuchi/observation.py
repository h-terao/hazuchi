from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import chex


class Observation(NamedTuple):
    """Summarize metrics."""

    metrics: dict[str, tuple[chex.Array, chex.Array]] | None = None

    @classmethod
    def create(
        cls, metrics: dict[str, chex.Array] | None = None, weight: float = 1.0
    ) -> Observation:
        if metrics is None:
            metrics = {}
        return cls(
            {key: (jax.lax.stop_gradient(val * weight), weight) for key, val in metrics.items()}
        )

    @property
    def accum_metrics(self):
        if self.metrics is None:
            return {}
        return self.metrics

    def keys(self):
        return self.accum_metrics.keys()

    def values(self):
        return self.accum_metrics.values()

    def items(self):
        return self.accum_metrics.items()

    @jax.jit
    def summary(self):
        return {key: jnp.sum(val) / jnp.sum(weight) for key, (val, weight) in self.items()}

    def scalar_summary(self, *, prefix: str | None = None, **scalars):
        if prefix is None:
            prefix = ""
        summary = {prefix + key: float(val) for key, val in self.summary().items()}
        summary = dict(summary, **scalars)
        return summary

    def add(self, metrics: dict[str, chex.Array], weight: float = 1.0) -> Observation:
        """Accumulate metrics."""
        return self + Observation.create(metrics, weight)

    def update(self, metrics: dict[str, chex.Array], weight: float = 1.0) -> Observation:
        """Overwrite metrics."""
        return self | Observation.create(metrics, weight)

    @jax.jit
    def __add__(self, other: Observation) -> Observation:
        updates = {}
        for key, (val, weight) in other.items():
            accum_val, accum_weight = self.accum_metrics.get(key, (0, 0))
            accum_val = jax.lax.stop_gradient(accum_val + val)
            accum_weight = jax.lax.stop_gradient(accum_weight + weight)
            updates[key] = (accum_val, accum_weight)
        accum_metrics = dict(self.accum_metrics, **updates)
        return Observation(accum_metrics)

    def __iadd__(self, other: Observation) -> Observation:
        return self + other

    def __or__(self, other: Observation) -> Observation:
        new_metrics = dict(self.accum_metrics, **other.accum_metrics)
        return Observation(new_metrics)

    def __ior__(self, other: Observation) -> Observation:
        return self | other

    @jax.jit
    def __mul__(self, other: float) -> Observation:
        new_metrics = {
            key: (jax.lax.stop_gradient(val * other), jax.lax.stop_gradient(weight * other))
            for key, (val, weight) in self.items()
        }
        return Observation(new_metrics)

    def __truediv__(self, other: float) -> Observation:
        return self * (1.0 / other)
