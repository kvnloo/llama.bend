"""No-model regressions for fail-closed oracle harnesses.

The oracles and child processes below are in-memory fixtures. These tests check
the harnesses, not model accuracy, and never write or replace model goldens.
"""
import contextlib
import io
import json
import runpy
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = "n1 q k v qr kr attn a n2 m y".split()
LAYER = "inv_freq 0\n" + "".join(f"{name} 0 0\n" for name in STAGES)
GEN = "prompt ids: 1\n  id 2  logit 1065353216 1.0  t=0.100s\ntext: p q\n"


def execute(script, stdout, returncode=0, mode=None):
    models = types.ModuleType("models")
    models.ROOT, models.D, models.T_DIR = str(ROOT), "/fixture-model", "/fixture-oracles"
    models.config = lambda: {"intermediate_size": 1}
    models.sizes = lambda: (1, 1, 1, 1, 1, 1)
    case = {"prompt": "p", "n": 1, "ids": [1], "gen": [2], "logits": [1.0], "text": "p q"}

    def fake_open(path, *args, **kwargs):
        path = str(path)
        if path.endswith("gen.json"):
            return io.StringIO(json.dumps([case]))
        if path.endswith("_ids.txt"):
            return io.StringIO("1")
        if "/bits/" in path:
            return io.BytesIO(b"fixture output\n")
        raise AssertionError(f"unexpected fixture read: {path}")

    def fake_run(command, **kwargs):
        out = stdout if kwargs.get("text") else stdout.encode()
        err = "fixture child failed" if kwargs.get("text") else b"fixture child failed"
        if kwargs.get("check") and returncode:
            raise subprocess.CalledProcessError(returncode, command, output=out, stderr=err)
        return subprocess.CompletedProcess(command, returncode, out, err)

    argv = [str(ROOT / "reference" / script), "fake-exe"]
    if mode:
        argv = [argv[0], mode, "fake-forward", "fake-gen"]
    output = io.StringIO()
    with patch.dict(sys.modules, {"models": models}), patch.object(sys, "argv", argv), \
            patch("builtins.open", fake_open), patch("subprocess.run", fake_run), \
            patch("os.makedirs"), patch("numpy.fromfile", lambda *a, **k: np.zeros(1, dtype="<f4")), \
            contextlib.redirect_stdout(output):
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exc:
            return exc.code
        except (subprocess.CalledProcessError, ValueError):
            return 1
    return 0


class ReferenceGuardTests(unittest.TestCase):
    def test_complete_fixtures_pass(self):
        for script, stdout, mode in [
            ("test_layer.py", LAYER, None), ("test_forward.py", "logits 0 0\n", None),
            ("test_gen.py", GEN, None), ("bits.py", "fixture output\n", "check"),
        ]:
            with self.subTest(script=script):
                self.assertEqual(execute(script, stdout, mode=mode), 0)

    def test_nonzero_children_fail_despite_matching_stdout(self):
        for script, stdout, mode in [
            ("test_layer.py", LAYER, None), ("test_forward.py", "logits 0 0\n", None),
            ("test_gen.py", GEN, None), ("bits.py", "fixture output\n", "check"),
        ]:
            with self.subTest(script=script):
                self.assertNotEqual(execute(script, stdout, returncode=1, mode=mode), 0)

    def test_every_layer_stage_required(self):
        for name in STAGES:
            with self.subTest(stage=name):
                self.assertNotEqual(execute("test_layer.py", LAYER.replace(f"{name} 0 0\n", "")), 0)

    def test_inv_freq_required_and_exact(self):
        for bad in ["", "inv_freq 1065353216\n", "inv_freq 0 0\n"]:
            with self.subTest(inv_freq=bad):
                self.assertNotEqual(execute("test_layer.py", LAYER.replace("inv_freq 0\n", bad)), 0)

    def test_layer_duplicate_rows_and_positions_fail(self):
        for row in ["n1 0 0\n", "inv_freq 0\n", "n1 1 0\n"]:
            with self.subTest(extra=row):
                self.assertNotEqual(execute("test_layer.py", LAYER + row), 0)

    def test_layer_nonfinite_or_wrong_width_fails(self):
        for row in ["n1 0 2143289344\n", "n1 0 2139095040\n", "n1 0 0 0\n"]:
            with self.subTest(row=row):
                self.assertNotEqual(execute("test_layer.py", LAYER.replace("n1 0 0\n", row)), 0)

    def test_forward_nonfinite_duplicates_and_extra_positions_fail(self):
        for stdout in ["logits 0 2143289344\n", "logits 0 2139095040\n",
                       "logits 0 0\nlogits 0 0\n", "logits 0 0\nlogits 1 0\n"]:
            with self.subTest(stdout=stdout):
                self.assertNotEqual(execute("test_forward.py", stdout), 0)

    def test_generation_nonfinite_or_missing_logit_fails(self):
        for stdout in [GEN.replace("1.0", "nan"), GEN.replace("1.0", "inf"),
                       GEN.replace("  id 2  logit 1065353216 1.0  t=0.100s\n", "")]:
            with self.subTest(stdout=stdout):
                self.assertNotEqual(execute("test_gen.py", stdout), 0)


if __name__ == "__main__":
    unittest.main()
