from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cvframes.iterate import IOCapture, iterate, iterate_sbs


@pytest.fixture
def video_capture():
    with patch("cv2.VideoCapture") as mock_capture:
        instance = mock_capture.return_value
        instance.isOpened.return_value = True
        instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8))
        ] * 5 + [(False, None)]
        yield instance


@pytest.fixture
def video_writer():
    with patch("cv2.VideoWriter") as mock_writer:
        yield mock_writer.return_value


def test_iocapture_init(video_capture, video_writer):
    cap = IOCapture("input.mp4", "output.mp4")
    assert cap.icap.isOpened.called
    assert cap.ocap is not None


def test_iocapture_read(video_capture):
    cap = IOCapture("input.mp4")
    ret, frame = cap.read()
    assert ret is True
    assert frame.shape == (480, 640, 3)


def test_iocapture_write(video_writer):
    cap = IOCapture("input.mp4", "output.mp4")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap.write(frame)
    cap.ocap.write.assert_called_once_with(frame)


def test_iocapture_release(video_capture, video_writer):
    cap = IOCapture("input.mp4", "output.mp4")
    cap.release()
    cap.icap.release.assert_called_once()
    cap.ocap.release.assert_called_once()


def test_iterate(video_capture):
    frames = list(iterate(Path("input.mp4")))
    assert len(frames) == 5
    assert frames[0].shape == (480, 640, 3)


def test_iterate_sbs(video_capture):
    frames = list(iterate_sbs(Path("input.mp4")))
    assert len(frames) == 5
    left, right = frames[0]
    assert left.shape == (480, 320, 3)
    assert right.shape == (480, 320, 3)
