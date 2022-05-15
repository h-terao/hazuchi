import chex


def hflip(img: chex.Array) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).

    Returns:
        Horizontally flipped images.
    """
    return img[..., :, ::-1, :]


def vflip(img: chex.Array) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).

    Returns:
        Vertically flipped images.
    """
    return img[..., ::-1, :, :]
