import asyncio

import click
import cv2

from cvframes.iterate import iterate


@click.group("cv-frames")
def main():
    pass


@main.command(name="show")
@click.argument("filepath", type=click.Path(exists=True))
def show(filepath):
    for i, (_, frame) in enumerate(iterate(filepath)):
        print("Frame:", i)
        cv2.imshow("cvframes-show", frame)

        # Wait for a key press and check if it's 'q' to exit
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


@main.command(name="server")
@click.option(
    "--http-port", default=8000, help="Port for HTTP/WebSocket server."
)
@click.option("--tcp-port", default=9999, help="Port for TCP frame source.")
def run(http_port, tcp_port):
    from cvframes.web.server import main

    asyncio.run(main(http_port, tcp_port))
