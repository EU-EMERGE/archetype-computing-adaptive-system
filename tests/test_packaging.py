"""Smoke tests for the installed package, without dataset downloads or training."""

import importlib
import importlib.util
from importlib.metadata import distribution
from pathlib import Path
import unittest

import torch

import acds


class InstalledPackageTests(unittest.TestCase):
    def setUp(self):
        self.raw = Path(acds.benchmarks.__file__).parent / "raw"

    def test_public_api_imports(self):
        for module in (acds.archetypes, acds.benchmarks):
            for name in module.__all__:
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(callable(getattr(module, name)))

    def test_distribution_metadata_and_license(self):
        installed = distribution("acds")
        self.assertEqual(installed.metadata["Requires-Python"], ">=3.11")
        self.assertEqual(installed.metadata["License-Expression"], "GPL-3.0-only")
        licenses = installed.metadata.get_all("License-File") or []
        self.assertIn("LICENSE", licenses)
        self.assertTrue(any(str(p).endswith("/licenses/LICENSE") for p in installed.files))

    def test_adiac_data_is_usable_after_installation(self):
        loaders = acds.benchmarks.get_adiac_data(
            self.raw / "adiac", bs_train=8, bs_test=8
        )
        self.assertEqual([len(loader.dataset) for loader in loaders], [270, 120, 391])
        for loader in loaders:
            samples, targets = next(iter(loader))
            self.assertEqual(samples.shape, (8, 176, 1))
            self.assertEqual(targets.shape, (8, 1))
            self.assertTrue(torch.isfinite(samples).all())

    def test_mackey_glass_data_is_usable_after_installation(self):
        datasets = acds.benchmarks.get_mackey_glass(self.raw, lag=10, washout=20)
        for samples, targets in datasets:
            self.assertEqual(samples.ndim, 1)
            self.assertEqual(len(samples) - len(targets), 20)
            self.assertGreater(len(targets), 0)
            self.assertTrue(torch.isfinite(samples).all())

    def test_mackey_glass_windows_have_required_dependencies(self):
        loaders = acds.benchmarks.get_mackey_glass_windows(
            self.raw, chunk_length=12, prediction_lag=1, tr_bs=8
        )
        samples, targets = next(iter(loaders[0]))
        self.assertEqual(samples.shape, (8, 12, 1))
        self.assertEqual(targets.shape, (8, 1))
        self.assertTrue(all(len(loader.dataset) > 0 for loader in loaders))

    def test_oscillator_forward_pass(self):
        model = acds.archetypes.RandomizedOscillatorsNetwork(
            n_inp=1, n_hid=8, dt=0.01, gamma=1.0, epsilon=1.0
        )
        states, final_state = model(torch.randn(2, 6, 1))
        self.assertEqual(states.shape, (2, 6, 8))
        self.assertEqual(final_state[0].shape, (2, 8))
        self.assertTrue(torch.isfinite(states).all())

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in (
            "gymnasium", "neat", "deap", "matplotlib", "PIL", "imageio", "graphviz"
        )),
        "Install acds[evolutionary] to test the optional evolutionary modules",
    )
    def test_evolutionary_extra_imports(self):
        for name in ("environment", "experiment", "utils", "run"):
            with self.subTest(module=name):
                importlib.import_module(f"acds.evolutionary.{name}")


if __name__ == "__main__":
    unittest.main()
