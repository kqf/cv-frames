import asyncio
import contextlib
import json

try:
    from aiohttp import web
except ImportError as e:
    missing = e.name if hasattr(e, "name") else "aiohttp"
    print(
        f"❌ Missing required dependency: {missing}\n"
        f"➡️  Please install it with:\n\n"
        f"   pip install cv-frames[web]\n"
    )
    raise SystemExit(1) from e

import importlib.resources as pkg_resources

MAPPING = {
    "ArrowUp": 82,
    "ArrowDown": 84,
    "ArrowLeft": 81,
    "ArrowRight": 83,
    "Escape": 27,
}


async def broadcast_task(
    frame_queue: asyncio.Queue,
    clients: set,
    clients_lock: asyncio.Lock,
):
    while True:
        frame = await frame_queue.get()
        async with clients_lock:
            to_remove = set()
            tasks = []
            for ws in clients:
                if ws.closed:
                    to_remove.add(ws)
                    continue

                async def send(ws):
                    try:
                        await ws.send_bytes(frame)
                    except Exception:
                        to_remove.add(ws)

                tasks.append(asyncio.create_task(send(ws)))
            await asyncio.gather(*tasks, return_exceptions=True)
            for ws in to_remove:
                clients.remove(ws)
        frame_queue.task_done()


async def handle_tcp_connection(
    reader,
    writer,
    frame_queue: asyncio.Queue,
):
    print("[TCP] Frame source connected")
    buffer = b""

    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                print("[TCP] Source disconnected")
                break
            buffer += chunk

            while b"\xff\xd8" in buffer and b"\xff\xd9" in buffer:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9") + 2
                if end <= start:
                    break
                jpg = buffer[start:end]
                buffer = buffer[end:]
                await frame_queue.put(jpg)

    except Exception as e:
        print(f"[TCP] Error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print("[TCP] Connection closed")


async def websocket_handler(request):
    app = request.app
    clients = app["clients"]
    clients_lock = app["clients_lock"]
    tcp_send_lock = app["tcp_send_lock"]
    tcp_conn_writer = app.get("tcp_conn_writer")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print(f"[WS] Client connected: {request.remote}")
    async with clients_lock:
        clients.add(ws)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "key":
                    key = data.get("key")
                    print(f"[WS] Key pressed: {key}")
                    code = MAPPING.get(
                        key, ord(key) if len(key) == 1 else None
                    )
                    if code is None:
                        continue

                    print(f"[WS] Sending key code: {code}")
                    async with tcp_send_lock:
                        if tcp_conn_writer := request.app.get(
                            "tcp_conn_writer"
                        ):
                            tcp_conn_writer.write(f"{code}".encode())
                            await tcp_conn_writer.drain()

            elif msg.type == web.WSMsgType.ERROR:
                print(f"[WS] Error: {ws.exception()}")
                break

    except Exception as e:
        print(f"[WS] Receive error: {e}")

    finally:
        async with clients_lock:
            clients.discard(ws)
        print(f"[WS] Client disconnected: {request.remote}")

    return ws


async def main(http_port: int, tcp_port: int):
    app = web.Application()
    frame_queue: asyncio.Queue = asyncio.Queue()
    clients: set[web.WebSocketResponse] = set()
    clients_lock = asyncio.Lock()
    tcp_send_lock = asyncio.Lock()
    tcp_conn_writer = None

    app["frame_queue"] = frame_queue
    app["clients"] = clients
    app["clients_lock"] = clients_lock
    app["tcp_send_lock"] = tcp_send_lock
    app["tcp_conn_writer"] = tcp_conn_writer

    # Get static paths relative to package
    index = pkg_resources.files("cvframes.web.static").joinpath("index.html")
    app.router.add_get("/", lambda request: web.FileResponse(str(index)))
    app.router.add_get("/ws", websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", http_port)
    await site.start()
    print(f"✅ Server running at: http://localhost:{http_port}")

    broadcaster = asyncio.create_task(
        broadcast_task(frame_queue, clients, clients_lock)
    )

    async def tcp_server_callback(reader, writer):
        app["tcp_conn_writer"] = writer
        await handle_tcp_connection(reader, writer, frame_queue)
        app["tcp_conn_writer"] = None

    server = await asyncio.start_server(
        tcp_server_callback, "0.0.0.0", tcp_port
    )
    print(f"[TCP] Waiting for frame source on port {tcp_port}")

    async with server:
        await server.serve_forever()

    broadcaster.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await broadcaster
