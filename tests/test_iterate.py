from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cvframes.iterate import iterate, iterate_sbs


@pytest.fixture
def video_capture():
    with ExitStack() as stack:
        stack.enter_context(
            patch("cvframes.iterate.IOCapture.isOpened", return_value=True)
        )
        stack.enter_context(
            patch(
                "cvframes.iterate.cv2.VideoCapture.__init__",
                return_value=None,
            )
        )
        mock_read = stack.enter_context(
            patch("cvframes.iterate.IOCapture.read")
        )
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
        # Path("output.mp4"),
    ],
)
def test_iterate(video_capture, opath):
    # sourcery skip: no-loop-in-tests
    for capture, frame in iterate(Path("input.mp4"), opath=opath):
        capture.write(frame)
        assert frame.shape == (480, 640, 3)


@pytest.mark.parametrize(
    "opath",
    [
        "",
        # Path("output.mp4"),
    ],
)
def test_iterate_sbs(video_capture, opath):
    # sourcery skip: no-loop-in-tests
    for capture, (lframe, rframe) in iterate_sbs(
        Path("input.mp4"), opath=opath
    ):
        capture.write(lframe)
        assert lframe.shape == (480, 320, 3)
        assert rframe.shape == (480, 320, 3)
