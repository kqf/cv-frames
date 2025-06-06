import importlib

try:
    importlib.import_module("aiohttp")
except ImportError as e:
    print(
        "❌ Missing required dependency: aiohttp\n"
        "➡️  Please install it with:\n\n"
        "   pip install cv-frames[web]\n"
    )
    raise SystemExit(1) from e

import importlib
import select
import socket
from contextlib import contextmanager

import cv2


class FrameStreamer:
    def __init__(self, sock):
        self.sock = sock

    def imshow(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        self.sock.sendall(buffer.tobytes())

    def waitKey(self, timeout_ms=1000):
        self.sock.setblocking(0)
        ready = select.select([self.sock], [], [], timeout_ms / 1000.0)
        if not ready[0]:
            return -1

        if data := self.sock.recv(1024):
            msg = data.decode().strip()
            return int(msg)
        return -1


@contextmanager
def connect_to_server(host="127.0.0.1", port=9999):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print("[Client] Connected to server")
    try:
        yield FrameStreamer(sock)
    finally:
        sock.close()
        print("[Client] Disconnected")
