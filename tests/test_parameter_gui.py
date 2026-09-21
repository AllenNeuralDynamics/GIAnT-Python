"""Headless checks for the parameter form's canonical split-option API."""

import math
import sys
import unittest
from unittest.mock import MagicMock, patch

from giant_python.gui.parameters_gui import run_parameter_gui
from giant_python.models.params import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
)


class TestParameterGui(unittest.TestCase):
    """Exercise form callbacks without creating a desktop window."""

    def setUp(self):
        """Replace toolkit widgets with controlled variables and callbacks."""
        self.tk = MagicMock()
        self.root = self.tk.Tk.return_value
        self.variables = []
        self.buttons = {}
        self.tk.StringVar.side_effect = self._string_var
        self.tk.BooleanVar.side_effect = self._boolean_var
        self.tk.ttk.Button.side_effect = self._button
        self.modules = patch.dict(sys.modules, {"tkinter": self.tk})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def _string_var(self, *, value):
        """Retain each field variable so simulated users can edit it."""
        variable = MagicMock()
        variable.get.return_value = value
        self.variables.append(variable)
        return variable

    def _boolean_var(self, *, value):
        """Track the ROI inclusion control separately from text fields."""
        self.roi_var = MagicMock()
        self.roi_var.get.return_value = value
        return self.roi_var

    def _button(self, parent, *, text, command):
        """Capture form actions instead of binding real toolkit events."""
        self.buttons[text] = command
        return MagicMock()

    def test_success_returns_fresh_split_options(self):
        """Scientific, execution, and annotation controls stay independent."""
        params = BandSiloParams(peakth=8)
        run = ExecutionOptions(max_workers=2, verbose=True, max_trials=4)
        roi = AnnotationOptions(interactive=False)

        def interact():
            """Simulate edits to a scientific field and both policy groups."""
            self.variables[0].get.return_value = "125"
            self.variables[8].get.return_value = "3"
            self.variables[9].get.return_value = "-2"
            self.roi_var.get.return_value = True
            self.buttons["OK"]()

        self.root.mainloop.side_effect = interact
        result = run_parameter_gui(params, execution=run, annotations=roi)
        assert result is not None
        science, execution, annotations = result
        self.assertEqual(science.analyze_hz, 125)
        self.assertEqual(science.peakth, 8)
        self.assertAlmostEqual(science.sparse_fac, math.exp(-2))
        labels = [
            call.kwargs["text"] for call in self.tk.ttk.Label.call_args_list
        ]
        self.assertNotIn("Operator:", labels)
        self.assertEqual(execution, ExecutionOptions(3, True, 4))
        self.assertEqual(annotations, AnnotationOptions(True, False))
        self.assertEqual(params.analyze_hz, 100)
        self.assertEqual(run.max_workers, 2)
        self.assertFalse(roi.enabled)
        self.assertFalse(roi.interactive)
        for supplied, returned in zip((params, run, roi), result):
            self.assertIsNot(supplied, returned)
        self.root.destroy.assert_called_once()
        self.tk.messagebox.showerror.assert_not_called()

    def test_cancel_returns_none(self):
        """Cancel does not manufacture a combined parameter result."""
        self.root.mainloop.side_effect = lambda: self.buttons["Cancel"]()
        self.assertIsNone(run_parameter_gui())
        self.root.destroy.assert_called_once()

    def test_invalid_input_shows_error_without_accepting(self):
        """Invalid scientific values leave the form open for correction."""

        def interact():
            """Try a nonpositive rate, then cancel the still-open form."""
            self.variables[0].get.return_value = "0"
            self.buttons["OK"]()
            self.root.destroy.assert_not_called()
            self.tk.messagebox.showerror.assert_called_once()
            self.buttons["Cancel"]()

        self.root.mainloop.side_effect = interact
        self.assertIsNone(run_parameter_gui())
        self.root.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
