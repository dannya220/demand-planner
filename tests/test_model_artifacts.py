import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.verify_models import verify_models


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "models" / "example.txt"
        self.model.parent.mkdir()
        self.model.write_text(
            "tree\nversion=v4\nfeature_names=lag_28d horizon\n"
            "tree_sizes=40 50\n\nTree=0\nTree=1\n"
        )
        self.manifest = self.root / "manifest.json"
        self.data = {
            "feature_columns": ["lag_28d", "horizon"],
            "models": [{"path": "models/example.txt", "trees": 2,
                        "bytes": self.model.stat().st_size,
                        "sha256": hashlib.sha256(self.model.read_bytes()).hexdigest()}],
        }
        self.manifest.write_text(json.dumps(self.data))

    def test_valid_model_is_accepted(self):
        self.assertEqual(verify_models(self.root, self.manifest), ["models/example.txt"])

    def test_changed_weights_are_rejected(self):
        self.model.write_text(self.model.read_text().replace("Tree=1", "Tree=2"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_models(self.root, self.manifest)

    def test_manifest_cannot_point_outside_root(self):
        self.data["models"][0]["path"] = "../example.txt"
        self.manifest.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            verify_models(self.root, self.manifest)

    def test_feature_order_is_checked(self):
        self.data["feature_columns"] = ["horizon", "lag_28d"]
        self.manifest.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Feature order"):
            verify_models(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
