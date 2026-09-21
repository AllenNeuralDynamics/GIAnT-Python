"""Existing tkinter BandSILo parameter form, with lazy toolkit imports."""

from dataclasses import asdict, replace
from typing import Optional

import numpy as np

from ..models.params import (
    AnnotationInput,
    AnnotationOptions,
    BandParamsInput,
    BandSiloParams,
    ExecutionInput,
    ExecutionOptions,
    resolve_band_options,
)

_PARAM_GUI_FIELDS = (
    ("analyze_hz", "Analyze Hz:"),
    ("decay_tau_s", "Decay Tau (s):"),
    ("baseline_window_s", "Baseline Window (s):"),
    ("denoise_window_s", "Denoise Window (s):"),
    ("vif", "Variance Inflation Factor (VIF):"),
    ("d_xy", "dXY:"),
    ("peakth", "Peak Threshold:"),
    ("peak_buffer", "Peak Buffer:"),
    ("max_workers", "Max Workers:"),
    ("operator", "Operator:"),
)


def run_parameter_gui(
    params: BandParamsInput = None,
    *,
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> Optional[  # pragma: no cover - interactive tkinter dialog
    tuple[BandSiloParams, ExecutionOptions, AnnotationOptions]
]:
    """Return split option objects, or None on cancellation, without mutation.

    Retains the original form and natural-log sparse-factor control. Toolkit
    code is imported only when this function is actually invoked.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    base, run, roi = resolve_band_options(params, execution, annotations)
    values = {**asdict(base), **asdict(run), **asdict(roi)}
    root = tk.Tk()
    root.title("SLAP2 Analysis Parameters")
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0)
    entries = {}
    row = 0
    for attr, label in _PARAM_GUI_FIELDS:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W)
        var = tk.StringVar(value=str(values[attr]))
        ttk.Entry(frame, textvariable=var, width=15).grid(row=row, column=1)
        entries[attr] = var
        row += 1
    ttk.Label(frame, text="Sparse Factor (log):").grid(
        row=row, column=0, sticky=tk.W
    )
    sparse_var = tk.StringVar(value=str(float(np.log(base.sparse_fac))))
    ttk.Entry(frame, textvariable=sparse_var, width=15).grid(row=row, column=1)
    row += 1
    ttk.Label(frame, text="Draw User ROIs?").grid(
        row=row, column=0, sticky=tk.W
    )
    draw_rois_var = tk.BooleanVar(value=roi.enabled)
    ttk.Checkbutton(frame, variable=draw_rois_var).grid(row=row, column=1)
    row += 1
    result: dict[
        str,
        Optional[tuple[BandSiloParams, ExecutionOptions, AnnotationOptions]],
    ] = {"options": None}

    def on_ok():
        """Validate controls and return a replacement parameter instance."""
        try:
            updates = {
                "analyze_hz": float(entries["analyze_hz"].get()),
                "decay_tau_s": float(entries["decay_tau_s"].get()),
                "baseline_window_s": float(entries["baseline_window_s"].get()),
                "denoise_window_s": float(entries["denoise_window_s"].get()),
                "vif": float(entries["vif"].get()),
                "d_xy": int(entries["d_xy"].get()),
                "peakth": float(entries["peakth"].get()),
                "peak_buffer": int(entries["peak_buffer"].get()),
                "sparse_fac": float(np.exp(float(sparse_var.get()))),
            }
            result["options"] = (
                replace(base, **updates),
                replace(run, max_workers=int(entries["max_workers"].get())),
                replace(
                    roi,
                    operator=entries["operator"].get(),
                    enabled=bool(draw_rois_var.get()),
                ),
            )
            root.destroy()
        except ValueError as error:
            messagebox.showerror("Error", f"Invalid parameter values: {error}")

    def on_cancel():
        """Discard edits and close the dialog."""
        root.destroy()

    button_frame = ttk.Frame(frame)
    button_frame.grid(row=row, column=0, columnspan=2, pady=10)
    ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
        side=tk.LEFT
    )
    root.grab_set()
    root.mainloop()
    return result["options"]
