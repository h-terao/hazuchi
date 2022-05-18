from __future__ import annotations
import jax.random as jrandom
import chex


def mixup(
    rng: chex.PRNGKey,
    img: chex.Array,
    label: chex.Array,
    img2: chex.Array | None = None,
    label2: chex.Array | None = None,
    *,
    beta: float = 0.5,
):
    """Mixup augmentation.

    Args:
        img2: If given, mix img and img2. Otherwise, shuffle img and use it instead of img2.
    """
    assert label.ndim > 1, "labels must be soft vector."
    assert (img2 is None and label2 is None) or (img2 is not None and label2 is not None)

    perm_rng, mix_rng = jrandom.split(rng)

    if img2 is None:
        batch_size = label.shape[0]
        perm = jrandom.permutation(perm_rng, batch_size)
        img2, label2 = img[perm], label[perm]

    v = jrandom.beta(mix_rng, beta, beta, dtype=img.dtype)

    new_images = v * img + (1 - v) * img2
    new_labels = v * label + (1 - v) * label2
    return new_images, new_labels
