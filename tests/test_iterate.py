from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cvframes.iterate import iterate, iterate_sbs


@pytest.fixture
def video_capture():
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    def _get(option):
        if option == cv2.CAP_PROP_FPS:
            return 30
        if option == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if option == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480

    with patch("cvframes.iterate.cv2.VideoCapture") as mock_capture:
        instance = mock_capture.return_value
        instance.isOpened.return_value = True
        instance.get.side_effect = _get
        instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8))
        ] * 5 + [(False, None)]
        yield instance


@pytest.fixture
def video_writer():
    with patch("cvframes.iterate.cv2.VideoWriter") as mock_writer:
        yield mock_writer.return_value


@pytest.mark.skip()
def test_iterate(video_capture):
    frames = list(iterate(Path("input.mp4")))
    assert len(frames) == 5
    assert frames[0].shape == (480, 640, 3)


def test_iterate_sbs(video_capture):
    # sourcery skip: no-loop-in-tests
    for lframe, rframe in iterate_sbs(Path("input.mp4")):
        assert lframe.shape == (480, 320, 3)
        assert rframe.shape == (480, 320, 3)
