import numpy as np

from dynamic_fusion.utils.datatypes import GrayImage, GrayImageFloat


def normalize(image: GrayImage) -> GrayImageFloat:
    image = image - np.min(image)
    return image / np.max(image)
