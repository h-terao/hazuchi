from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

__all__ = ["shift"]


def _left_shift(x: chex.Array, num_shift: int, shift_axis: int, fill_value: float = 0):
    # [1, 2, 3] -> [1, 2, 3, p] -> [2, 3, p]
    pad_width = [[0, 0] if i != shift_axis else [0, num_shift] for i in range(x.ndim)]
    new_x = jnp.pad(x, pad_width, cval=fill_value)
    new_x = jax.lax.dynamic_slice(
        new_x, [0 if i != shift_axis else num_shift for i in range(x.ndim)], x.shape
    )
    return new_x


def _right_shift(x: chex.Array, num_shift: int, shift_axis: int, fill_value: float = 0):
    # [1, 2, 3] -> [p, 1, 2, 3] -> [p, 1, 2]
    pad_width = [[0, 0] if i != shift_axis else [num_shift, 0] for i in range(x.ndim)]
    new_x = jnp.pad(x, pad_width, cval=fill_value)
    new_x = jax.lax.dynamic_slice(new_x, [0 for _ in range(x.ndim)], x.shape)
    return new_x


def shift(
    x: chex.Array,
    num_channels: int,
    shift_axis: int,
    channel_axis: int,
    num_shift: int = 1,
    direction: str = "both",
    fill_value: float = 0,
):
    """Shift columns of arrays specified as shift_axis.

    Args:
        x (array): An input array to shift.
        num_channels (int): Total number of channels to shift.
                            If direction is `both`, half of num_channels arrays are shifted to left.
                            and other half of num_channels arrays are shifted to right.
        shift_axis (int): Axis to shift.
        channel_axis (int): Axis of channels.
        num_shift (int): Number of pixels to shift.
        direction (str): Direction to shift. "both", "left" or "right".
        fill_value (float): Padding value for the edge of shifted arrays.
    """
    if shift_axis < 0:
        shift_axis += x.ndim

    if channel_axis < 0:
        channel_axis += x.ndim

    if direction == "both":
        num_channels = num_channels // 2
        left_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 for _ in range(x.ndim)],
            slice_sizes=[s if i != channel_axis else num_channels for i, s in enumerate(x.shape)],
        )
        right_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 if i != channel_axis else num_channels for i in range(x.ndim)],
            slice_sizes=[s if i != channel_axis else num_channels for i, s in enumerate(x.shape)],
        )
        center_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 if i != channel_axis else 2 * num_channels for i in range(x.ndim)],
            slice_sizes=[
                s if i != channel_axis else s - 2 * num_channels for i, s in enumerate(x.shape)
            ],
        )

        left_array = _left_shift(left_array, num_shift, shift_axis, fill_value)
        right_array = _right_shift(right_array, num_shift, shift_axis, fill_value)
        out = jnp.concatenate([left_array, right_array, center_array], axis=channel_axis)

    elif direction == "left":
        left_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 for _ in range(x.ndim)],
            slice_sizes=[s if i != channel_axis else num_channels for i, s in enumerate(x.shape)],
        )
        center_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 if i != channel_axis else num_channels for i in range(x.ndim)],
            slice_sizes=[
                s if i != channel_axis else s - num_channels for i, s in enumerate(x.shape)
            ],
        )
        left_array = _left_shift(left_array, num_shift, shift_axis, fill_value)
        out = jnp.concatenate([left_array, center_array], axis=channel_axis)

    elif direction == "right":
        right_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 for _ in range(x.ndim)],
            slice_sizes=[s if i != channel_axis else num_channels for i, s in enumerate(x.shape)],
        )
        center_array = jax.lax.dynamic_slice(
            x,
            start_indices=[0 if i != channel_axis else num_channels for i in range(x.ndim)],
            slice_sizes=[
                s if i != channel_axis else s - num_channels for i, s in enumerate(x.shape)
            ],
        )
        right_array = _right_shift(left_array, num_shift, shift_axis, fill_value)
        out = jnp.concatenate([left_array, center_array], axis=channel_axis)

    else:
        raise ValueError("direction must be both, left or right.")

    return out
