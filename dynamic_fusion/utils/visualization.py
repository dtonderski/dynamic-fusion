import numpy as np


def create_red_blue_cmap(N) -> np.ndarray:  # type: ignore
    assert N % 2 == 1
    cm = np.zeros([N, 3])
    cm[: N // 2, 0] = np.sqrt(np.linspace(N / 2, 1, N // 2) * 2 / N)
    cm[N // 2 :, 2] = np.sqrt(np.linspace(1, N / 2, N - N // 2) * 2 / N)
    cm[N // 2, :] = 0.5
    return cm


def img_to_colormap(img, cmap, clims=None) -> np.ndarray:  # type: ignore
    if clims is None:
        clims = np.array([-1, 1]) * np.max(np.abs(img))
    N, C = cmap.shape
    # assert N % 2 == 1
    grid_x = np.linspace(clims[0], clims[1], N)
    img_colored = np.zeros(img.shape + (C,))
    for i in range(C):  # 3 color channels
        img_colored[..., i] = np.interp(img, grid_x, cmap[:, i])
    return img_colored
