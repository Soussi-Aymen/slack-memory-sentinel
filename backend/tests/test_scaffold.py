"""Scaffold smoke tests: the package is importable and the container mounts are in place."""

from pathlib import Path

import pytest

import contextdrift


def test_package_is_installed_and_importable():
    assert contextdrift.__version__


@pytest.mark.skipif(not Path("/app").is_dir(), reason="only meaningful inside the container")
@pytest.mark.parametrize("mount", ["/app/data", "/app/frontend"])
def test_read_only_mounts_are_present(mount):
    assert Path(mount).is_dir(), f"{mount} is not mounted"
