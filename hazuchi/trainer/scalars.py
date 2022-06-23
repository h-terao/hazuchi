from __future__ import annotations

import jax


@jax.jit
def accumulate_scalars(accum_scalars, new_scalars, weight):
    updates = {}
    for key, scalar in new_scalars.items():
        accum_scalar, accum_weight = accum_scalars.get(key, (0, 0))
        accum_scalar += scalar.mean() * weight
        accum_weight += weight
        updates[key] = (accum_scalar, accum_weight)
    return dict(accum_scalars, **updates)


def summarize_scalars(prefix, accum_scalars):
    return {prefix + key: float(val / weight) for key, (val, weight) in accum_scalars.items()}
