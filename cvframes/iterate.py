from pathlib import Path
from typing import Callable, Generator, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np

T = TypeVar("T")


class IOCapture:
    def __init__(self, source: Union[str, Path], oname: Union[str, Path] = ""):
        self.icap = cv2.VideoCapture(str(source))
        self.ocap = (
            cv2.VideoWriter(
                str(oname),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.icap.get(cv2.CAP_PROP_FPS),
                (
                    int(self.icap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.icap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
            if oname
            else None
        )

    def is_opened(self) -> bool:
        return self.icap.isOpened()

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.icap.read()

    def write(self, frame: np.ndarray) -> None:
        if self.ocap is not None:
            self.ocap.write(frame)

    def release(self) -> None:
        self.icap.release()
        if self.ocap is not None:
            self.ocap.release()

    def set(self, prop_id: int, value: float) -> None:
        self.icap.set(prop_id, value)


def iterate_generic(
    ipath: Path,
    opath: Optional[Path],
    start_frame: int,
    stop_frame: int,
    process_frames: Callable[[np.ndarray], T],
) -> Generator[Tuple[IOCapture, T], None, None]:
    capture = IOCapture(str(ipath), oname=opath or "")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = start_frame

    if not capture.is_opened():
        raise RuntimeError(f"Cannot open video file: {ipath}")

    try:
        while True:
            ret, frame = capture.read()
            count += 1
            if not ret:
                break
            if stop_frame > 0 and count >= stop_frame:
                break

            yield capture, process_frames(frame)
    finally:
        capture.release()


def iterate(
    ipath: Path,
    opath: Optional[Path] = None,
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[Tuple[IOCapture, np.ndarray], None, None]:
    return iterate_generic(
        ipath,
        opath,
        start_frame,
        stop_frame,
        lambda frame: frame,
    )


def iterate_sbs(
    ipath: Path,
    opath: Optional[Path] = None,
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[Tuple[IOCapture, Tuple[np.ndarray, np.ndarray]], None, None]:
    def processor(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, width, _ = frame.shape
        mid = width // 2
        return frame[:, :mid, :], frame[:, mid:, :]

    return iterate_generic(
        ipath,
        opath,
        start_frame,
        stop_frame,
        processor,
    )
