import jax


def try_copy(params, pretrained_params):
    """Copy pretrained_params as much as possible.
    This function is useful to load the pretrained parameters to the other initialized parameters
    for transfer learning.
    """
    return jax.tree_util.tree_map(
        lambda x: (x[1] if x[0].shape == x[1].shape else x[0]),
        params,
        pretrained_params,
    )
