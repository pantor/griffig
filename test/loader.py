from pathlib import Path

import cv2
import numpy as np

from griffig import OrthographicImage


class Loader:
    data_path = Path(__file__).parent / 'data'

    @classmethod
    def get_image(cls, episode_id: str):
        mat_color = cv2.imread(str(cls.data_path / f'{episode_id}-rc.jpeg'), cv2.IMREAD_UNCHANGED).astype(np.uint16) * 255
        mat_depth = cv2.imread(str(cls.data_path / f'{episode_id}-rd.jpeg'), cv2.IMREAD_UNCHANGED).astype(np.uint16) * 255
        mat_depth = cv2.cvtColor(mat_depth, cv2.COLOR_BGR2GRAY)

        mat = np.concatenate([mat_color, np.expand_dims(mat_depth, axis=2)], axis=2)
        return OrthographicImage(mat, 2000.0, 0.22, 0.41)
