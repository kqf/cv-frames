from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


def video_data(
    ipath: Path,
    opath: Path | str = "",
    start_frame: int = -1,
    stop_frame: int = -1,
):
    capture = cv2.VideoCapture(str(ipath))
    capture.icap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = start_frame
    try:
        while True:
            ret, frame = capture.read()
            count += 1
            if not ret:
                break
            if stop_frame > 0 and count >= stop_frame:
                break
            yield capture, frame
    finally:
        capture.release()


def iterate_sbs(
    ipath: Path,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    capture = cv2.VideoCapture(str(ipath))

    if not capture.isOpened():
        print(f"Error: Could not open video {ipath}")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            break  # Properly exit loop when the video ends

        height, width, _ = frame.shape
        mid = width // 2
        limage = frame[:, :mid, :]
        rimage = frame[:, mid:, :]

        yield limage, rimage

    capture.release()
