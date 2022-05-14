from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct
import chex


@struct.dataclass
class Observation:
    """An immutable class to summarize metrics.

    Attributes:
        values (dict of str: tuple[array, array]): Accumulated metrics to summarize.
    """

    values: dict[str, tuple[chex.Array, chex.Array]] = struct.field(True, default_factory=dict)

    @classmethod
    def create(cls, metrics: dict[str, chex.Array] | None = None, weight: float = 1.0) -> Observation:
        """Create a new observation.
        Args:
            metrics (dict of str: array): Metrics to summarize.
            weight (float, optional): Weight of metrics for accumulation.
                                      Maybe, batch size is usually passed.
        Returns:
            Observation: Returns a new observation initialized by the given metrics.
        """
        if metrics is None:
            return cls()
        else:
            values = jax.tree_map(lambda v: (v * weight, weight), metrics)
            return cls(values)

    def add(self, metrics: dict[str, chex.Array], weight: float = 1.0) -> Observation:
        """Accumulate metrics."""
        return self + Observation.create(metrics, weight)

    def update(self, metrics: dict[str, chex.Array], weight: float = 1.0) -> Observation:
        """Overwrite metrics."""
        return self | Observation.create(metrics, weight)

    def summary(self, **kwargs) -> dict[str, chex.Array]:
        """Summarize metrics.
        Args:
            **kwargs: Values to overwrite a summary.
                      Useful to add current steps, epochs, and elapsed time into the summary.
        """
        summary = {key: jnp.sum(val) / jnp.sum(weight) for key, (val, weight) in self.values.items()}
        return dict(summary, **kwargs)

    def scalar_summary(self, *, prefix: str | None = None, **kwargs) -> dict[str, float]:
        """Returns a summary."""
        if prefix is None:
            prefix = ""
        summary = {f"{prefix}{key}": float(val) for key, val in self.summary().items()}
        return dict(summary, **kwargs)

    def __add__(self, other: Observation) -> Observation:
        updates = {}
        for key, (val, weight) in other.values.items():
            accum_val, accum_weight = self.values.get(key, (0, 0))
            accum_val += val
            accum_weight += weight
            updates[key] = (accum_val, accum_weight)
        values = dict(self.values, **updates)
        return Observation(values)

    def __iadd__(self, other: Observation) -> Observation:
        return self + other

    def __or__(self, other: Observation) -> Observation:
        values = dict(self.values, **other.values)
        return Observation(values)

    def __ior__(self, other: Observation) -> Observation:
        return self | other

    def __mul__(self, other: float) -> Observation:
        return Observation({key: (val * other, weight * other) for key, (val, weight) in self.values.items()})

    def __truediv__(self, other: float) -> Observation:
        return Observation({key: (val / other, weight / other) for key, (val, weight) in self.values.items()})
