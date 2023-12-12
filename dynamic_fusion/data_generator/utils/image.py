import numpy as np

from dynamic_fusion.utils.datatypes import GrayImage


def normalize(image: GrayImage) -> GrayImage:
    image = image - np.min(image)
    return image / np.max(image)
