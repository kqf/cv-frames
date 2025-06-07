cv-frames
=========

.. image:: https://github.com/kqf/cv-frames/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/kqf/cv-frames/actions
   :alt: Tests

.. image:: https://img.shields.io/pypi/dm/cv-frames.svg
   :target: https://pypi.org/project/cv-frames/
   :alt: PyPI Downloads

``cv-frames`` is a lightweight utility that simplifies working with video files frame-by-frame using OpenCV.
It includes both a Python API and a command-line interface (CLI) for interacting with video frames in an intuitive way.

----

Installation
------------

Install via pip:

.. code-block:: bash

    pip install cv-frames

----

Command Line Usage
------------------

After installation, you can use the CLI:

.. code-block:: bash

    cv-frames show path/to/video.mp4

Navigate through frames with any key. Press **q** to quit.

----

Python Usage
------------

Example: Read and show frames programmatically

.. code-block:: python

    from pathlib import Path
    from cvframes import iterate

    for i, (_, frame) in enumerate(iterate(Path("video.mp4"))):
        print(f"Frame {i} shape:", frame.shape)

Example: Split side-by-side video into left/right frames

.. code-block:: python

    from cvframes import iterate_sbs

    # Write the processed frames to disc
    for capture, (left, right) in iterate_sbs("sbs_video.mp4", oname="processed.mp4"):
        processed = do_something(left)
        capture.write(processed)
