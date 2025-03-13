from pathlib import Path
from typing import Callable, Generator, Optional, Tuple, TypeVar

import cv2
import numpy as np

T = TypeVar("T")


class IOCapture(cv2.VideoCapture):
    def __init__(self, *args, oname: str | Path = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.ocap = (
            cv2.VideoWriter(
                str(oname),
                cv2.VideoWriter_fourcc(*"mp4v"),
                super().get(cv2.CAP_PROP_FPS),
                (
                    int(super().get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(super().get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
            if oname
            else None
        )

    def write(self, frame):
        if self.ocap is not None:
            self.ocap.write(frame)

    def release(self):
        super().release()
        if self.ocap is not None:
            self.ocap.release()


def iterate_generic(
    ipath: Path,
    opath: Optional[Path],
    start_frame: int,
    stop_frame: int,
    process_frames: Callable[[np.ndarray], T],
) -> Generator[T, None, None]:
    capture = IOCapture(str(ipath), oname=opath or "")
    capture.icap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = start_frame

    if not capture.isOpened():
        print(f"Error: Could not open video {ipath}")
        return
    print("->")

    try:
        while True:
            ret, frame = capture.read()
            count += 1
            if not ret:
                break
            if stop_frame > 0 and count >= stop_frame:
                break

            yield process_frames(frame)
    finally:
        capture.release()


def iterate(
    ipath: Path,
    opath: Optional[Path] = None,
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[np.ndarray, None, None]:
    def processor(frame: np.ndarray) -> np.ndarray:
        return frame

    return iterate_generic(
        ipath,
        opath,
        start_frame,
        stop_frame,
        processor,
    )


def iterate_sbs(
    ipath: Path,
    opath: Optional[Path] = None,
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    def processor(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, width, _ = frame.shape
        mid = width // 2
        limage = frame[:, :mid, :]
        rimage = frame[:, mid:, :]
        return limage, rimage

    return iterate_generic(
        ipath,
        opath,
        start_frame,
        stop_frame,
        processor,
    )
