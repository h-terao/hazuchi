from __future__ import annotations
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import chex


@dataclass(frozen=True)
class Observation:
    """Summarize metrics."""

    accum_metrics: dict[str, chex.Array] = field(default_factory=dict)
    accum_weights: dict[str, chex.Array] = field(default_factory=dict)

    @classmethod
    def create(cls, metrics: dict[str, chex.Array] | None = None, weight: float = 1.0) -> Observation:
        if metrics is None:
            metrics = {}
        accum_metrics = {key: val * weight for key, val in metrics.items()}
        accum_weights = {key: weight for key in metrics}
        return cls(accum_metrics, accum_weights)

    def keys(self):
        return list(self.accum_metrics)

    def summary(self):
        return jax.tree_map(
            lambda val, weight: jnp.mean(val) / jnp.mean(weight),
            self.accum_metrics,
            self.accum_weights,
        )

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

    def __add__(self, other: Observation) -> Observation:
        metric_updates = {}
        weight_updates = {}
        for key in other.keys():
            metric = other.accum_metrics[key]
            weight = other.accum_weights[key]

            metric_updates[key] = self.accum_metrics.get(key, 0) + metric
            weight_updates[key] = self.accum_weights.get(key, 0) + weight

        new_metrics = dict(self.accum_metrics, **metric_updates)
        new_weights = dict(self.accum_weights, **weight_updates)
        return Observation(new_metrics, new_weights)

    def __iadd__(self, other: Observation) -> Observation:
        return self + other

    def __or__(self, other: Observation) -> Observation:
        new_metrics = dict(self.accum_metrics, **other.accum_metrics)
        new_weights = dict(self.accum_weights, **other.accum_weights)
        return Observation(new_metrics, new_weights)

    def __ior__(self, other: Observation) -> Observation:
        return self | other

    def __mul__(self, other: float) -> Observation:
        new_metrics = jax.tree_map(lambda x: x * other, self.accum_metrics)
        new_weights = jax.tree_map(lambda x: x * other, self.accum_weights)
        return Observation(new_metrics, new_weights)

    def __truediv__(self, other: float) -> Observation:
        new_metrics = jax.tree_map(lambda x: x / other, self.accum_metrics)
        new_weights = jax.tree_map(lambda x: x / other, self.accum_weights)
        return Observation(new_metrics, new_weights)
