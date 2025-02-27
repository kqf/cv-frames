from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


class IOCapture(cv2.VideoCapture):
    def __init__(self, iname: str, oname: str = ""):
        self.icap = cv2.VideoCapture(iname)
        self.ocap = (
            cv2.VideoWriter(
                oname,
                cv2.VideoWriter_fourcc(*"H264"),
                self.icap.get(cv2.CAP_PROP_FPS),
                (
                    int(self.icap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.icap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
            if oname
            else None
        )

    def read(self):
        return self.icap.read()

    def write(self, frame):
        if self.ocap is not None:
            self.ocap.write(frame)

    def release(self):
        self.icap.release()
        if self.ocap is not None:
            self.ocap.release()


def video_data(
    ipath: Path,
    opath: Path | str = "",
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[Tuple[cv2.VideoCapture, np.ndarray], None, None]:
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
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    capture = cv2.VideoCapture(str(ipath))
    capture.icap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = start_frame

    if not capture.isOpened():
        print(f"Error: Could not open video {ipath}")
        return

    while True:
        ret, frame = capture.read()
        count += 1
        if not ret:
            break
        if stop_frame > 0 and count >= stop_frame:
            break

        height, width, _ = frame.shape
        mid = width // 2
        limage = frame[:, :mid, :]
        rimage = frame[:, mid:, :]

        yield limage, rimage

    capture.release()
