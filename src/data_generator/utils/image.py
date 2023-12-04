import numpy as np

from on_the_fly.trainers.utils.datatypes import GrayImage


def normalize(image: GrayImage) -> GrayImage:
    image = image - np.min(image)
    return image / np.max(image)
