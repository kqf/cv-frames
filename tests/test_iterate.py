from pathlib import Path

import cv2
import numpy as np
import pytest

from cvframes.iterate import iterate, iterate_sbs


def count_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0

    while True:
        ret, _ = capture.read()
        if not ret:
            break
        count += 1

    capture.release()
    return count


@pytest.fixture
def nframes() -> int:
    return 5


@pytest.fixture
def shape() -> tuple[int, int]:
    return 640, 480


@pytest.fixture
def video(tmp_path: Path, nframes: int, shape: tuple[int, int]) -> Path:
    ipath = tmp_path / "input.mp4"
    width, height = shape
    ivideo = cv2.VideoWriter(
        str(ipath),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        shape,
    )

    for _ in range(nframes):
        ivideo.write(np.zeros((height, width, 3), dtype=np.uint8))
    ivideo.release()
    return ipath


@pytest.fixture(params=[False, True], ids=["no-output", "with-output"])
def opath(request: pytest.FixtureRequest, tmp_path: Path) -> Path | None:
    return tmp_path / "output.mp4" if request.param else None


def test_iterate(
    video: Path, opath: Path | None, nframes: int, shape: tuple[int, int]
):
    # sourcery skip: no-loop-in-tests
    width, height = shape
    seen = 0

    for capture, frame in iterate(video, opath=opath):
        capture.write(frame)
        assert frame.shape == (height, width, 3)
        seen += 1

    assert seen == nframes


def test_iterate_writes_every_frame(video: Path, tmp_path: Path, nframes: int):
    # sourcery skip: no-loop-in-tests
    opath = tmp_path / "output.mp4"

    for capture, frame in iterate(video, opath=opath):
        capture.write(frame)

    assert count_frames(opath) == nframes


@pytest.mark.parametrize(
    "start_frame, stop_frame, expected",
    [
        (-1, -1, 5),
        (2, -1, 3),
        (0, 5, 5),
        (0, 3, 3),
        (2, 4, 2),
    ],
)
def test_iterate_range(
    video: Path, start_frame: int, stop_frame: int, expected: int
):
    frames = list(
        iterate(video, start_frame=start_frame, stop_frame=stop_frame)
    )

    assert len(frames) == expected


def test_iterate_sbs(
    video: Path, opath: Path | None, nframes: int, shape: tuple[int, int]
):
    # sourcery skip: no-loop-in-tests
    width, height = shape
    seen = 0

    for capture, (lframe, rframe) in iterate_sbs(video, opath=opath):
        capture.write(lframe)
        assert lframe.shape == (height, width // 2, 3)
        assert rframe.shape == (height, width // 2, 3)
        seen += 1

    assert seen == nframes
