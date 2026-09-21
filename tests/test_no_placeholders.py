"""Prevent removed placeholder modules, APIs, and empty functions returning."""

import ast
import inspect
import unittest
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from textwrap import dedent

from giant_python.models import TrialTable
from giant_python.pipeline import Pipeline

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "giant_python"

REMOVED_MODULES = (
    "numerics.deconv",
    "numerics.nmf",
    "numerics.registration",
    "numerics.variance",
    "pipeline.base",
    "pipeline.organize",
    "pipeline.register",
    "gui.draw_rois",
    "models.alignment",
)
REMOVED_STAGES = (
    "Stage",
    "TrialTableBuilder",
    "MotionCorrector",
    "MultiRoiRegistration",
    "StripRegistration",
    "BandRegistration",
    "build_trial_table",
    "build_trial_table_slap2",
    "verify_files",
    "multi_roi_registration",
    "strip_registration",
    "band_registration",
    "silo",
)
REMOVED_KERNELS = (
    "deconvolve_trace",
    "nmf_decompose",
    "xcorr2_nans_weighted",
    "xcorr2_nans",
    "xcorr2_nans3d",
    "dft_register_clipped",
    "activity_image",
    "interp_frame",
)
REMOVED_IO = (
    "save_struct_h5",
    "get_online_motion",
    "scanimagetiff_wrapper",
    "scanimagetiff_data_wrapper",
)
REMOVED_EXPORTS = {
    "giant_python": (
        *REMOVED_STAGES,
        *REMOVED_KERNELS,
        *REMOVED_IO,
        "AlignParams",
        "AlignmentData",
        "DrawROIs",
    ),
    "giant_python.models": ("AlignParams", "AlignmentData"),
    "giant_python.models.params": ("AlignParams",),
    "giant_python.pipeline": REMOVED_STAGES,
    "giant_python.pipeline.pipeline": (*REMOVED_STAGES, "AlignParams"),
    "giant_python.pipeline.extract": ("silo",),
    "giant_python.numerics": REMOVED_KERNELS,
    "giant_python.numerics.interpolation": ("interp_frame",),
    "giant_python.io": REMOVED_IO,
    "giant_python.io.hdf5": ("save_struct_h5",),
    "giant_python.io.slap2": ("get_online_motion",),
    "giant_python.io.tiff": (
        "scanimagetiff_wrapper",
        "scanimagetiff_data_wrapper",
    ),
    "giant_python.gui": ("DrawROIs", "annotate_rois"),
}


def _is_stub_function(node):
    """Recognize functions containing only a docstring, pass, or ellipsis."""
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    return all(
        isinstance(statement, ast.Pass)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        )
        for statement in body
    )


def _placeholder_violations(tree, relative_path):
    """Return line-numbered AST violations, exempting one exact protocol."""
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    violations = []
    for node in ast.walk(tree):
        if (
            (isinstance(node, ast.Name) and node.id == "NotImplementedError")
            or (
                isinstance(node, ast.Attribute)
                and node.attr == "NotImplementedError"
            )
            or (
                isinstance(node, ast.alias)
                and node.name == "NotImplementedError"
            )
        ):
            violations.append((node.lineno, "NotImplementedError"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_stub_function(node):
                parent = parents[node]
                is_reader_contract = (
                    relative_path == "io/slap2.py"
                    and isinstance(parent, ast.ClassDef)
                    and parent.name == "Slap2Reader"
                    and isinstance(parents[parent], ast.Module)
                    and any(
                        isinstance(base, ast.Name) and base.id == "Protocol"
                        for base in parent.bases
                    )
                    and node.name == "getLineData"
                    and len(node.body) == 1
                    and ast.get_docstring(node) is not None
                )
                if not is_reader_contract:
                    violations.append((node.lineno, "stub: " + node.name))
    return violations


class TestRemovedPlaceholders(unittest.TestCase):
    """Removal means physical absence and no discoverable replacement API."""

    def test_deleted_modules_are_physically_absent(self):
        """Neither source files nor same-name package shells may remain."""
        for module in REMOVED_MODULES:
            path = PACKAGE_ROOT.joinpath(*module.split("."))
            with self.subTest(module=module):
                self.assertFalse(path.with_suffix(".py").exists())
                self.assertFalse(path.exists())

    def test_deleted_modules_cannot_be_discovered(self):
        """Import discovery cannot find old modules or compatibility shims."""
        for module in REMOVED_MODULES:
            with self.subTest(module=module):
                self.assertIsNone(find_spec("giant_python." + module))

    def test_removed_owner_and_package_exports_are_absent(self):
        """Removed symbols cannot survive in owners, exports, or completion."""
        for module_name, names in REMOVED_EXPORTS.items():
            module = import_module(module_name)
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertFalse(hasattr(module, name))
                    self.assertNotIn(name, getattr(module, "__all__", ()))
                    self.assertNotIn(name, dir(module))

    def test_removed_model_and_pipeline_members_are_absent(self):
        """Future serialization, stages, and unused session state are gone."""
        for owner in (TrialTable, TrialTable()):
            with self.subTest(owner=owner):
                self.assertFalse(hasattr(owner, "to_h5"))
        for owner in (Pipeline, Pipeline()):
            for name in (
                "from_config",
                "organize",
                "register",
                "run_all",
                "save_dir",
                "align_params",
            ):
                with self.subTest(owner=owner, name=name):
                    self.assertFalse(hasattr(owner, name))

    def test_constructor_exposes_only_supported_options(self):
        """Science and routing remain distinct from keyword-only policies."""
        parameters = inspect.signature(Pipeline).parameters
        self.assertEqual(
            list(parameters),
            ["microscope", "params", "execution", "annotations", "scan_mode"],
        )
        for name in ("microscope", "params"):
            self.assertEqual(
                parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        for name in ("execution", "annotations", "scan_mode"):
            self.assertEqual(
                parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
            )

    def test_removed_constructor_keywords_are_rejected(self):
        """Direct and loaded-table constructors reject removed options."""
        table = TrialTable()
        for name in ("save_dir", "align_params", "silo_params"):
            with self.subTest(keyword=name):
                with self.assertRaisesRegex(TypeError, name):
                    Pipeline(**{name: None})
                with self.assertRaisesRegex(TypeError, name):
                    Pipeline.from_trial_table(table, **{name: None})

    def test_production_ast_contains_no_placeholders(self):
        """Scan every source function, including private and nested helpers."""
        paths = sorted(PACKAGE_ROOT.rglob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            with self.subTest(path=relative):
                self.assertEqual(_placeholder_violations(tree, relative), [])

    def test_real_reader_protocol_is_retained(self):
        """The used raw-reader typing declaration is not a removed stub API."""
        from giant_python.io import Slap2Reader
        from giant_python.io.slap2 import Slap2Reader as Reader

        self.assertIs(Slap2Reader, Reader)
        self.assertTrue(callable(Reader.getLineData))
        self.assertIn("getLineData", Reader.__dict__)
        tree = ast.parse(
            (PACKAGE_ROOT / "io" / "slap2.py").read_text(encoding="utf-8-sig")
        )
        reader = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Slap2Reader"
        )
        method = next(
            node
            for node in reader.body
            if isinstance(node, ast.FunctionDef) and node.name == "getLineData"
        )
        self.assertTrue(_is_stub_function(method))
        self.assertEqual(len(method.body), 1)
        self.assertIsNotNone(ast.get_docstring(method))
        self.assertEqual(_placeholder_violations(tree, "io/slap2.py"), [])


class TestPlaceholderGuard(unittest.TestCase):
    """Exercise the AST guard so forbidden cases cannot silently escape it."""

    def test_empty_function_variants_are_detected(self):
        """Docstrings, pass, ellipsis, async methods, and nested stubs fail."""
        cases = (
            ("def stub(): pass", "stub"),
            ("def stub(): ...", "stub"),
            ('def stub(): "Only a docstring."', "stub"),
            ('def stub():\n    "Description."\n    pass\n    ...', "stub"),
            ("async def stub(): ...", "stub"),
            ("class Example:\n    def method(self): pass", "method"),
            ("def outer():\n    def inner(): ...", "inner"),
        )
        for source, name in cases:
            with self.subTest(source=source):
                violations = _placeholder_violations(
                    ast.parse(source), "example.py"
                )
                self.assertEqual(
                    [reason for _, reason in violations], ["stub: " + name]
                )

    def test_not_implemented_error_forms_are_detected(self):
        """Bare, constructed, qualified, and aliased error references fail."""
        for source in (
            "raise NotImplementedError",
            "raise NotImplementedError('future')",
            "raise builtins.NotImplementedError()",
            "from builtins import NotImplementedError as FutureError",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    _placeholder_violations(ast.parse(source), "example.py"),
                    [(1, "NotImplementedError")],
                )

    def test_real_functions_and_non_code_mentions_are_allowed(self):
        """Ignore comments, strings, and pass within implemented functions."""
        source = dedent('''\
            # NotImplementedError is mentioned, not raised.
            def implemented(value):
                """NotImplementedError is just documentation here."""
                try:
                    return value[...]
                except IndexError:
                    pass
                return None

            async def asynchronous():
                return "NotImplementedError"
            ''')
        self.assertEqual(
            _placeholder_violations(ast.parse(source), "example.py"), []
        )

    def test_protocol_exception_is_exact(self):
        """No other file, class, method, or empty body gets an exemption."""
        declaration = dedent('''\
            class Slap2Reader(Protocol):
                def getLineData(self):
                    """Vendor reader typing contract."""
            ''')
        self.assertEqual(
            _placeholder_violations(ast.parse(declaration), "io/slap2.py"), []
        )
        cases = (
            (declaration, "other.py"),
            (declaration.replace("Slap2Reader", "OtherReader"), "io/slap2.py"),
            (declaration.replace("getLineData", "other"), "io/slap2.py"),
            (declaration.replace("(Protocol)", ""), "io/slap2.py"),
            (
                declaration.replace(
                    '"""Vendor reader typing contract."""', "pass"
                ),
                "io/slap2.py",
            ),
            (declaration + "        ...\n", "io/slap2.py"),
        )
        for source, path in cases:
            with self.subTest(source=source, path=path):
                self.assertEqual(
                    len(_placeholder_violations(ast.parse(source), path)), 1
                )


if __name__ == "__main__":
    unittest.main()
