import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe_data', ROOT / 'scripts/prepare_probe_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProbeDataTests(unittest.TestCase):
    def test_small_bundle_disjoint_deterministic_and_no_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / 'source'
            (root / 'train').mkdir(parents=True)
            for i in range(6):
                for suffix in ('_sat.jpg', '_mask.png'):
                    (root / 'train' / f'{i}{suffix}').write_bytes(str(i).encode())
            train, val = base / 'train.json', base / 'val.json'
            train.write_text(json.dumps([f'train/{i}_sat.jpg' for i in range(4)]))
            val.write_text(json.dumps([f'train/{i}_sat.jpg' for i in (4, 5)]))
            first = module.prepare(root, train, val, base / 'one', 2, 1)
            second = module.prepare(root, train, val, base / 'two', 2, 1)
            self.assertEqual(first, second)
            self.assertEqual(len(first['files']), 6)
            self.assertFalse((base / 'one/splits/test.json').exists())
            self.assertEqual(len(list((base / 'one/data/train').iterdir())), 6)
            with self.assertRaises(FileExistsError):
                module.prepare(root, train, val, base / 'one', 2, 1)
            val.write_text(train.read_text())
            with self.assertRaisesRegex(ValueError, 'overlap'):
                module.prepare(root, train, val, base / 'bad', 2, 1)
            self.assertFalse((base / 'bad').exists())

    def test_escape_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / 'manifest.json'
            path.write_text(json.dumps(['../secret_sat.jpg']))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                module.select_entries(path, 1, 42, base)


if __name__ == '__main__':
    unittest.main()
