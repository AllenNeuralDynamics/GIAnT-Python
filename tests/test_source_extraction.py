"""Tests for the source_extraction subpackage."""

import unittest

from giant_python.source_extraction.sel_act import sel_act


class TestSELAct(unittest.TestCase):
    """Tests for sel_act."""

    def test_not_implemented(self):
        """Test that sel_act raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            sel_act()


if __name__ == "__main__":
    unittest.main()
