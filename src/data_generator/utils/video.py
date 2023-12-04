import numpy as np

from on_the_fly.trainers.utils.datatypes import GrayVideo


def normalize(video: GrayVideo) -> GrayVideo:
    video = video - np.min(video)
    return video / np.max(video)
