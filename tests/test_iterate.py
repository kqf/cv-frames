import pytest

from cvframes.iterate import iterate_sbs


@pytest.mark.skip
def test_sbs():
    for frame in iterate_sbs("dummy"):
        print(frame.shape)
