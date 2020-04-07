from torchvision import transforms


def _DCGAN_get_value(x):
    """Helper function for `DCGAN_transform`."""
    if isinstance(x, (int, float)):
        return (x, x, x)
    elif isinstance(x, (tuple, list)):
        if len(x) == 3:
            return x
        else:
            raise ValueError("Must be tuple/list of length 3. Got {} of "
                             "length {} instead.".format(type(x), len(x)))
    else:
        raise ValueError(
            "Invalid argument type. Expected float, int or tuple/list of "
            "length 3. Got {} instead.".format(type(x)))


def DCGAN_transform(image_size=64, mean=0.0, std=1.0):
    """Return an image transformer as described in the DCGAN paper, which
    resizes, center-crops and normalizes input image to the range [-1, 1]
    corresponding to the range of the output of the Tanh function.
        By default `mean=0.0` and `std=1.0`, meaning that the images will be
        normalized to the range [-1, 1], corresponding to the range of the
        output of the Tanh function, as described in the DCGAN paper:
            Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised
            Representation Learning with Deep Convolutional Generative
            Adversarial Networks.
        Set `mean=0.5` and `std=0.5` to use the transformation as implemented
        in the PyTorch tutorial on DCGAN, which normalizes to the range [0, 1]:
            https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    Parameters
    ----------
    image_size : int
        Image height and width of generator's output and discriminator's input.
    mean : float or tuple of size 3
        If `float`, the same mean will be applied to all three image channels.
        If `tuple` of size 3, each value will correspond to an image channel.
    std : float or tuple of size 3
        If `float`, the same standard deviation will be applied to all three
        image channels.
        If `tuple` of size 3, each value will correspond to an image channel.
    """
    mean = _DCGAN_get_value(mean)
    std = _DCGAN_get_value(std)
    transformer = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transformer
