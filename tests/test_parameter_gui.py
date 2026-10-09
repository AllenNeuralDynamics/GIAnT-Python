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
        self.boolean_variables = []
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
        """Track checkbox controls separately from text fields."""
        self.roi_var = MagicMock()
        self.roi_var.get.return_value = value
        self.boolean_variables.append(self.roi_var)
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
            self.variables[9].get.return_value = "2.5"
            self.variables[10].get.return_value = "-2"
            self.boolean_variables[0].get.return_value = True
            self.roi_var.get.return_value = True
            self.buttons["OK"]()

        self.root.mainloop.side_effect = interact
        result = run_parameter_gui(params, execution=run, annotations=roi)
        assert result is not None
        science, execution, annotations = result
        self.assertEqual(science.analyze_hz, 125)
        self.assertEqual(science.peakth, 8)
        self.assertEqual(science.z_tol, 2.5)
        for name in ("simple_trace_extraction", "convolution_trace_extraction"):
            self.assertFalse(hasattr(science, name))
            self.assertFalse(hasattr(params, name))
        self.assertTrue(science.trace_debias)
        self.assertFalse(params.trace_debias)
        self.assertEqual(params.z_tol, 1.5)
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

    def test_removed_trace_flags_are_rejected_before_opening_form(self):
        """Removed dictionary keys cannot enter the GUI's base configuration."""
        for name in ("simple_trace_extraction", "convolution_trace_extraction"):
            for value in (False, True):
                with self.subTest(name=name, value=value):
                    supplied = {name: value}
                    with self.assertRaisesRegex(TypeError, name):
                        run_parameter_gui(supplied)
                    self.assertEqual(supplied, {name: value})
        self.tk.Tk.assert_not_called()
        self.root.mainloop.assert_not_called()

    def test_convolution_controls_preserve_trace_settings(self):
        """Accepting the GUI retains the independent trace-only options."""
        params = BandSiloParams(
            decay_tau_s=.3,
            trace_beta_fraction=.02, trace_max_iter=900,
            trace_tolerance=1e-9, trace_debias=True, trace_event_threshold=.001,
        )
        self.root.mainloop.side_effect = lambda: self.buttons["OK"]()
        result = run_parameter_gui(params)
        self.assertEqual(result[0], params)
        self.assertIsNot(result[0], params)
        self.assertEqual(result[0].decay_tau_s, .3)

    def test_trace_mode_toggles_are_replaced_by_always_saved_label(self):
        """Only debias and ROI checkboxes remain; extraction is not optional."""
        self.root.mainloop.side_effect = lambda: self.buttons["OK"]()
        result = run_parameter_gui()
        labels = [call.kwargs["text"] for call in self.tk.ttk.Label.call_args_list]
        self.assertIn("LS and convolution traces are always saved.", labels)
        self.assertEqual(len(self.boolean_variables), 2)
        self.assertEqual(self.tk.ttk.Checkbutton.call_count, 2)
        self.assertEqual(
            [label for label in labels if label.endswith("?")],
            ["Debias Convolution Traces?", "Draw User ROIs?"],
        )
        for call, variable in zip(
            self.tk.ttk.Checkbutton.call_args_list, self.boolean_variables
        ):
            self.assertIs(call.kwargs["variable"], variable)
        for name in ("simple_trace_extraction", "convolution_trace_extraction"):
            self.assertFalse(hasattr(result[0], name))
        self.assertFalse(result[2].enabled)
        self.tk.messagebox.showerror.assert_not_called()

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
