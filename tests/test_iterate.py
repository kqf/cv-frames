from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cvframes.iterate import iterate, iterate_sbs


@pytest.fixture
def video_capture():
    with patch(
        "cvframes.iterate.IOCapture.isOpened", return_value=True
    ), patch("cvframes.iterate.IOCapture.read") as mock_read:
        mock_read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        ] * 5 + [(False, None)]

        yield


def video_writer():
    with patch("cvframes.iterate.cv2.VideoWriter") as mock_writer:
        yield mock_writer.return_value


@pytest.mark.parametrize(
    "opath",
    [
        "",
        # "output.mp4",
    ],
)
def test_iterate(video_capture, opath):
    frames = list(iterate(Path("input.mp4"), opath=Path(opath)))
    assert len(frames) == 5
    assert frames[0].shape == (480, 640, 3)


@pytest.mark.skip
@pytest.mark.parametrize(
    "opath",
    [
        "",
        "output.mp4",
    ],
)
def test_iterate_sbs(video_capture, opath):
    # sourcery skip: no-loop-in-tests
    for lframe, rframe in iterate_sbs(Path("input.mp4"), opath=Path(opath)):
        assert lframe.shape == (480, 320, 3)
        assert rframe.shape == (480, 320, 3)
