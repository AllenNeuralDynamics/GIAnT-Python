"""Canonical kernel ownership, dependency direction, and retired APIs."""

import ast
import importlib
import importlib.util
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "giant_python"

# Each kernel has one implementation, in its canonical owning module.
OWNERS = {
    "numerics.baseline": (
        "_movmean_nan",
        "_compute_f0_column",
        "compute_f0",
        "assemble_dff",
    ),
    "numerics.peaks": (
        "gaussian_peaks_integrated",
        "_gaussian_peaks_integrated_val_jac",
        "_lsq_curvefit",
        "_fit_isotropic_then_anisotropic",
        "_make_bounds",
        "_peak_mask",
        "_buffer_mask",
        "_process_cc_new_peak",
        "_refine_residual_peaks",
        "detect_peaks_2d",
        "get_act_im_peaks",
    ),
    "numerics.interpolation": ("nearest_interp",),
    "numerics.morphology": ("fast_dilation",),
    "progress": ("log", "progress"),
    "extraction.band.motion_binning": (
        "bin_motion",
        "select_motion_bins",
        "bin_motion_yx",
    ),
    "extraction.band.geometry": (
        "ref_pixs_to_drc",
        "build_subsample_matrix_inds",
        "_pad_to",
        "build_combined_psf",
        "threshold_and_crop_psf",
        "build_selected_pixel_mask",
        "pixel_coords_from_idxs",
        "selected_pixels_2d_for_plane",
    ),
    "extraction.band.operators": (
        "build_sparse_h",
        "expand_psf",
        "shrink_psf",
        "gaussian_kernel_2d",
        "build_convolution_matrix",
        "build_motion_h_matrices",
    ),
    "extraction.band.inputs": (
        "load_lookup_table",
        "_ref_stack_group",
        "find_reference_file",
        "load_reference_stack",
        "default_psf",
        "load_psf",
    ),
    "extraction.band.background": (
        "_collect_interp_columns",
        "_interp_columns",
        "build_interp_data",
        "baseline_window_frames",
        "compute_rolling_baseline",
        "assemble_background",
        "fit_noise_variance_model",
        "compute_residual",
    ),
    "extraction.band.activity": (
        "_plane_column_maps",
        "_valid_sel_cols_for_motion",
        "compute_rho",
        "mask_high_nan_rho",
        "decay_kernel_1d",
        "smooth_rho",
    ),
    "extraction.band.localization": (
        "sel_pix_gaussian_profile",
        "sel_pix_patch_profile",
        "init_source_params",
        "build_a_patches",
        "build_x_support_mots",
        "project_spatial_profiles",
        "_motion_frames",
        "solve_phi_motion",
        "fit_phi_all_motions",
        "multiplicative_nmf",
        "fit_gaussian_profiles",
        "variance_sortorder",
        "reorder_sources",
        "compute_source_snr",
        "fit_sources",
    ),
    "extraction.band.summary_images": (
        "compute_mean_image",
        "accumulate_activity_image",
        "finalize_activity_image",
        "phase_matched_nanmedian",
        "snake_aligned_nanmedian",
    ),
    "extraction.band.traces": (
        "compute_high_res_traces",
        "get_high_res_traces",
    ),
    "extraction.band.trial_data": (
        "compute_ds_frames",
        "accumulate_superpixel_data",
        "read_band_trial_data",
        "assemble_lowres_data",
        "save_lowres_data",
        "load_lowres_data",
    ),
}


def _source_tree(module):
    """Parse a package module without importing its dependencies.

    Parameters
    ----------
    module : str
        Module path relative to ``giant_python``.

    Returns
    -------
    ast.Module
        Parsed source tree.
    """
    path = PACKAGE_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def _imports(module, tree):
    """Resolve static import targets, including function-local imports.

    Parameters
    ----------
    module : str
        Module path relative to ``giant_python``.
    tree : ast.Module
        Parsed source tree.

    Yields
    ------
    str
        Absolute imported module or member path.
    """
    parent = module.rpartition(".")[0]
    package = "giant_python" + ("." + parent if parent else "")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            target = importlib.util.resolve_name(name, package)
            for alias in node.names:
                yield target + "." + alias.name


class TestNumericArchitecture(unittest.TestCase):
    """Guard canonical ownership and low-level import boundaries."""

    def test_retired_namespaces_have_no_python_sources(self):
        """Permit orphan caches, but never active modules or empty stubs."""
        for namespace in ("bandsilo", "math"):
            with self.subTest(namespace=namespace):
                self.assertEqual(
                    list((PACKAGE_ROOT / namespace).rglob("*.py")), []
                )

    def test_production_imports_do_not_reference_retired_namespaces(self):
        """Inspect imports, including relative/local ones, not docstrings."""
        retired = {"giant_python.bandsilo", "giant_python.math"}
        for path in PACKAGE_ROOT.rglob("*.py"):
            module = ".".join(
                path.relative_to(PACKAGE_ROOT).with_suffix("").parts
            )
            for target in _imports(module, _source_tree(module)):
                with self.subTest(module=module, target=target):
                    self.assertFalse(
                        any(
                            target == name or target.startswith(name + ".")
                            for name in retired
                        )
                    )

    def test_public_exports_do_not_restore_retired_namespaces(self):
        """Public exports cannot expose the removed compatibility APIs."""
        for name in (
            "giant_python",
            "giant_python.numerics",
            "giant_python.extraction",
            "giant_python.extraction.band",
        ):
            namespace = importlib.import_module(name)
            for retired in ("bandsilo", "math"):
                with self.subTest(namespace=name, retired=retired):
                    self.assertNotIn(
                        retired, getattr(namespace, "__all__", ())
                    )

    def test_shared_numerics_do_not_import_application_layers(self):
        """Shared kernels cannot depend on acquisition, IO, UI, or backends."""
        forbidden = {"h5py", "tifffile", "slap2_utils", "tkinter", "napari"}
        for path in (PACKAGE_ROOT / "numerics").glob("*.py"):
            module = "numerics." + path.stem
            for target in _imports(module, _source_tree(module)):
                with self.subTest(module=module, target=target):
                    self.assertNotIn(target.split(".")[0], forbidden)
                    self.assertTrue(
                        not target.startswith("giant_python.")
                        or target.startswith("giant_python.numerics.")
                    )

    def test_band_kernels_do_not_depend_on_legacy_or_workflows(self):
        """Keep workflows above kernels and acquisition at input boundaries."""
        blocked = (
            "giant_python.bandsilo.",
            "giant_python.math.",
            "giant_python.gui.",
            "giant_python.pipeline.",
            "giant_python.extraction.band.workflow.",
        )
        input_boundaries = {
            "extraction.band.inputs",
            "extraction.band.trial_data",
            "extraction.band.traces",
        }
        for module in OWNERS:
            if not module.startswith("extraction.band."):
                continue
            for target in _imports(module, _source_tree(module)):
                with self.subTest(module=module, target=target):
                    self.assertFalse(target.startswith(blocked))
                    if module not in input_boundaries:
                        self.assertFalse(target.startswith("giant_python.io."))
                        self.assertFalse(
                            target.startswith(
                                "giant_python.extraction.band.trial_data."
                            )
                        )

    def test_namespaces_do_not_eagerly_import_backends(self):
        """Importing a namespace must not itself import IO or workflows."""
        for module in (
            "numerics.__init__",
            "extraction.__init__",
            "extraction.band.__init__",
        ):
            with self.subTest(module=module):
                for node in _source_tree(module).body:
                    self.assertNotIsInstance(
                        node, (ast.Import, ast.ImportFrom)
                    )

    def test_shared_helpers_and_summary_image_dependency(self):
        """Acquisition-independent helpers live in numerics, not trial IO."""
        dependencies = {
            "extraction.band.summary_images": (
                "giant_python.numerics.morphology.fast_dilation",
            ),
            "extraction.band.traces": (
                "giant_python.numerics.interpolation.nearest_interp",
            ),
            "extraction.band.trial_data": (
                "giant_python.io.slap2.open_slap2_file",
            ),
        }
        for module, targets in dependencies.items():
            imports = set(_imports(module, _source_tree(module)))
            for target in targets:
                with self.subTest(module=module, target=target):
                    self.assertIn(target, imports)

    def test_kernels_have_one_actual_definition(self):
        """Scan all production modules, not just the expected owner list."""
        definitions = {}
        for path in PACKAGE_ROOT.rglob("*.py"):
            module = ".".join(
                path.relative_to(PACKAGE_ROOT).with_suffix("").parts
            )
            for node in _source_tree(module).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.setdefault(node.name, []).append(module)
        for owner, names in OWNERS.items():
            for name in names:
                with self.subTest(owner=owner, name=name):
                    self.assertEqual(definitions.get(name), [owner])
