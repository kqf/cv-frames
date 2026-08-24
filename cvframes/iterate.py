from collections.abc import Callable, Generator
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

T = TypeVar("T")


class IOCapture:
    def __init__(
        self,
        source: str | Path,
        oname: str | Path = "",
        oshape: tuple[int, int] | None = None,
    ):
        self.icap = cv2.VideoCapture(str(source))
        self.oname = str(oname) if oname else None
        self.oshape = oshape
        self._ocap: cv2.VideoWriter | None = None

    @property
    def ocap(self) -> cv2.VideoWriter | None:
        if self.oname is None:
            return None

        if self.oshape is None:
            return None

        if self._ocap is None:
            self._ocap = cv2.VideoWriter(
                self.oname,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.icap.get(cv2.CAP_PROP_FPS),
                self.oshape,
            )

            if not self._ocap.isOpened():
                raise RuntimeError(
                    f"Cannot open output video file: {self.oname}"
                )

        return self._ocap

    def is_opened(self) -> bool:
        return self.icap.isOpened()

    def read(self) -> tuple[bool, np.ndarray]:
        return self.icap.read()

    def write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        shape = (width, height)

        if self.oshape is None:
            self.oshape = shape
            return

        if shape != self.oshape:
            raise ValueError(
                "All output frames must have the same resolution. "
                f"Expected {self.oshape[0]}x{self.oshape[1]}, "
                f"got {width}x{height}."
            )

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
    opath: Path | None,
    start_frame: int,
    stop_frame: int,
    process_frames: Callable[[np.ndarray], T],
) -> Generator[tuple[IOCapture, T], None, None]:
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
    opath: Path | None = None,
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[tuple[IOCapture, np.ndarray], None, None]:
    return iterate_generic(
        ipath,
        opath,
        start_frame,
        stop_frame,
        lambda frame: frame,
    )


def iterate_sbs(
    ipath: Path,
    opath: Path | None = None,
    start_frame: int = -1,
    stop_frame: int = -1,
) -> Generator[tuple[IOCapture, tuple[np.ndarray, np.ndarray]], None, None]:
    def processor(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
