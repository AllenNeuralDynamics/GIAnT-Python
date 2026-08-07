"""Tests for giant_python.bandsilo.progress (opt-in logging + bars)."""

import contextlib
import io
import unittest

from giant_python.bandsilo.progress import log, progress


class TestLog(unittest.TestCase):
    """log prints only when verbose is set."""

    def test_verbose_prints(self):
        """A verbose log writes the message to stdout."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            log("hello", True)
        self.assertEqual(buf.getvalue().strip(), "hello")

    def test_quiet_is_silent(self):
        """A non-verbose log writes nothing."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            log("hello", False)
        self.assertEqual(buf.getvalue(), "")


class TestProgress(unittest.TestCase):
    """progress wraps an iterable only when verbose is set."""

    def test_quiet_returns_same_object(self):
        """Without verbose, the iterable is returned unchanged."""
        r = range(3)
        self.assertIs(progress(r, desc="x", verbose=False), r)

    def test_verbose_wraps_but_preserves_items(self):
        """With verbose, a tqdm wrapper still yields the same items."""
        out = progress(range(3), desc="x", verbose=True)
        self.assertIsNot(out, range(3))
        self.assertEqual(list(out), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
