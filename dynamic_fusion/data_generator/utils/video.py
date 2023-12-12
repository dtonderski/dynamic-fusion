import numpy as np

from dynamic_fusion.utils.datatypes import GrayVideo


def normalize(video: GrayVideo) -> GrayVideo:
    video = video - np.min(video)
    return video / np.max(video)
