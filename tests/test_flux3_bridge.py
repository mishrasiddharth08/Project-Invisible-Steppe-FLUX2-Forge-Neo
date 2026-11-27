import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Flux3BridgeTests(unittest.TestCase):
    def check_layout(self, layout):
        name = "steppe_flux3_bridge_test"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / "steppe_flux3_bridge.py"
        )
        steppe = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            spec.loader.exec_module(steppe)

        class AutoEncoder:
            def decode(self, *args):
                return "original"

        runtime = types.SimpleNamespace(generate=lambda request: steppe._active.get())
        comfy = types.SimpleNamespace(generate=lambda request: steppe._active.get())
        engine = comfy if layout == "experimental" else runtime
        bridge = types.SimpleNamespace(generate=lambda enabled: engine.generate({"seed": 19}))
        imports = {
            "flux3_forge.runtime": runtime,
            "flux3_forge.bridge": bridge,
            "flux3_forge.vendor.flux3.autoencoder": types.SimpleNamespace(AutoEncoder=AutoEncoder),
        }
        if layout == "experimental":
            imports["flux3_forge.comfy_bridge"] = comfy

        def load(name):
            if name not in imports:
                raise ModuleNotFoundError(name="nested_dependency" if layout == "broken" else name)
            return imports[name]

        def callback(*args):
            pass

        original = (bridge.generate, runtime.generate, comfy.generate, AutoEncoder.decode)
        with patch.object(steppe.importlib, "import_module", side_effect=load):
            self.assertTrue(steppe.install(lambda enabled: callback if enabled else None))
            try:
                self.assertIsNone(bridge.generate(False))
                if layout == "broken":
                    with self.assertRaises(ModuleNotFoundError) as raised:
                        bridge.generate(True)
                    self.assertEqual(raised.exception.name, "nested_dependency")
                else:
                    self.assertEqual(bridge.generate(True), (callback, 19))
                self.assertIsNone(steppe._active.get())
                self.assertEqual((runtime.generate, comfy.generate, AutoEncoder.decode), original[1:])
            finally:
                steppe.uninstall()
        self.assertIs(bridge.generate, original[0])

    def test_default_runtime_has_no_comfy_bridge(self):
        self.check_layout("default")

    def test_experimental_runtime_still_uses_comfy_bridge(self):
        self.check_layout("experimental")

    def test_missing_dependency_is_not_silenced(self):
        self.check_layout("broken")


if __name__ == "__main__":
    unittest.main()
