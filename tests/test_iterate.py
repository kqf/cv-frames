from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from cvframes.iterate import iterate, iterate_sbs


@pytest.fixture
def video(tmp_path: Path):
    opath = tmp_path / "test_video.mp4"
    ovideo = cv2.VideoWriter(
        str(opath),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (640, 480),
    )

    for _ in range(5):
        ovideo.write(np.zeros((480, 640, 3), dtype=np.uint8))
    ovideo.release()
    return opath


@pytest.mark.parametrize(
    "opath",
    [
        None,
        Path("output.mp4"),
    ],
)
def test_iterate(video: Path, opath: Optional[Path]):
    # sourcery skip: no-loop-in-tests
    for capture, frame in iterate(video, opath=opath):
        capture.write(frame)
        assert frame.shape == (480, 640, 3)


@pytest.mark.skip("skipping")
@pytest.mark.parametrize(
    "opath",
    [
        None,
        Path("output.mp4"),
    ],
)
def test_iterate_sbs(video: Path, opath: Optional[Path]):
    # sourcery skip: no-loop-in-tests
    for capture, (lframe, rframe) in iterate_sbs(
        Path("input.mp4"), opath=opath
    ):
        capture.write(lframe)
        assert lframe.shape == (480, 320, 3)
        assert rframe.shape == (480, 320, 3)
