from pathlib import Path
import cv2

from typing import Generator, Tuple
import numpy as np


def iterate_sbs(
    path: Path,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        print(f"Error: Could not open video {path}")
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
