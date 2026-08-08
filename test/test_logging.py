import logging

import pytest

from pyxdf import load_xdf

logger = logging.getLogger("pyxdf.pyxdf")


@pytest.fixture(autouse=True)
def restore_level():
    """Guard the test session against the very leak these tests check for."""
    level = logger.level
    yield
    logger.setLevel(level)


@pytest.mark.parametrize("verbose", [None, True, False])
def test_verbose_does_not_leak_log_level(tmp_path, verbose):
    """The level set by verbose is scoped to the call, including on failure."""
    logger.setLevel(logging.NOTSET)

    with pytest.raises(Exception, match="does not exist"):
        load_xdf(tmp_path / "missing.xdf", verbose=verbose)

    assert logger.level == logging.NOTSET


def test_verbose_none_leaves_the_logger_untouched(tmp_path, monkeypatch):
    """With verbose=None the level is never set, so nothing is restored over it."""
    calls = []
    monkeypatch.setattr(logger, "setLevel", calls.append)

    with pytest.raises(Exception, match="does not exist"):
        load_xdf(tmp_path / "missing.xdf", verbose=None)

    assert calls == []


def test_verbose_does_not_override_application_level(tmp_path):
    """A level the application set is still in place after a call."""
    logger.setLevel(logging.ERROR)

    with pytest.raises(Exception, match="does not exist"):
        load_xdf(tmp_path / "missing.xdf", verbose=True)

    assert logger.level == logging.ERROR


@pytest.mark.parametrize(("verbose", "expected"), [(True, True), (False, False)])
def test_verbose_applies_during_the_call(tmp_path, caplog, verbose, expected):
    """verbose still decides whether the import message is emitted."""
    logger.setLevel(logging.NOTSET)
    # Capture whatever the logger lets through, whichever level pytest was invoked with
    caplog.set_level(logging.NOTSET)

    with pytest.raises(Exception, match="does not exist"):
        load_xdf(tmp_path / "missing.xdf", verbose=verbose)

    assert ("Importing XDF file" in caplog.text) is expected
