"""scripts/probe_3090.py 的有界迭代、配置/划分护栏与 CPU 编排测试。

全部使用假引擎，不加载 SAM 权重、不读取真实 DeepGlobe 图像、不启动训练。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_3090.py"
_spec = importlib.util.spec_from_file_location("probe_3090_under_test", SCRIPT)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"无法加载 {SCRIPT}")
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe
_spec.loader.exec_module(probe)

from diffpeak.inference import DEFAULT_CONFIG  # noqa: E402


class _FakeScaler:
    def __init__(self) -> None:
        self.scale = 1.0

    def is_enabled(self) -> bool:
        return False

    def get_scale(self) -> float:
        return self.scale


class _FakeEngine:
    instances = []
    fail_in = None

    def __init__(self, config, device=None, config_dir=None) -> None:
        self.config = copy.deepcopy(config)
        self.train_cfg = self.config["train"]
        self.accumulation_steps = int(self.train_cfg["accumulation_steps"])
        self.update_epochs = int(self.config["rl"]["update_epochs"])
        self.group_size = int(self.config["rl"]["group_size"])
        self.rl_algorithm = "grpo"
        self.device = torch.device("cpu")
        self.scaler = _FakeScaler()
        self.global_step = 0
        self.scheduler_stage_lr = 0.0
        self.stage = None
        self.prepare_calls = []
        self.epoch_calls = []
        self.init_calls = []
        self.validate_calls = 0
        _FakeEngine.instances.append(self)

    def _trainable_parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]

    def _prepare_stage(self, stage, epochs, steps_per_epoch) -> None:
        self.stage = stage
        self.scheduler_stage_lr = float(self.config["stages"][stage]["lr"])
        self.prepare_calls.append((stage, epochs, steps_per_epoch))

    def _build_train_loader(self, epoch):
        return [{"index": index, "epoch": epoch} for index in range(3)]

    def _train_epoch_supervised(self, stage, epoch, loader):
        if getattr(loader, "max_items", 0) == 0:
            return {"loss": 1.0, "optimizer_steps": 0.0, "skipped_groups": 0.0}
        if _FakeEngine.fail_in == "measure" and getattr(
            loader, "phase", ""
        ) == "measurement":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        items = list(loader)
        self.epoch_calls.append(("supervised", stage, epoch, len(items)))
        return {
            "loss": 1.0,
            "seg": 0.5,
            "aux": float("nan"),
            "optimizer_steps": float(
                math.ceil(len(items) / self.accumulation_steps)
            ),
            "skipped_groups": 0.0,
        }

    def _train_epoch_grpo(self, stage, epoch, loader):
        if getattr(loader, "max_items", 0) == 0:
            return {
                "reward": float("nan"),
                "optimizer_steps": 0.0,
                "skipped_update_steps": 0.0,
                "skipped_groups": 0.0,
            }
        if _FakeEngine.fail_in == "measure" and getattr(
            loader, "phase", ""
        ) == "measurement":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        items = list(loader)
        self.epoch_calls.append(("grpo", stage, epoch, len(items)))
        return {
            "reward": 0.7,
            "grpo": -0.1,
            "iou": 0.2,
            "optimizer_steps": float(len(items)),
            "skipped_update_steps": 0.0,
            "skipped_groups": 0.0,
        }

    def _resolved_config(self):
        return copy.deepcopy(self.config)

    def _model_states(self):
        states = {
            "sam": {"weight": torch.zeros(2)},
            "condition_encoder": {},
            "unet": {"weight": torch.ones(1)},
            "diffusion": {},
        }
        if self.stage == "D":
            states["ref_unet"] = {"weight": torch.ones(1)}
        return states

    def _save_payload(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), path)

    def load_init_weights(self, path):
        self.init_calls.append(str(path))

    def validate(self, stage):
        if _FakeEngine.fail_in == "validate":
            raise RuntimeError("simulated validation failure")
        self.validate_calls += 1
        return {"iou": 0.1, "score": 0.1}


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"strict JSON 含非有限字面量：{token}")

    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_reject), text


class ProbeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        _FakeEngine.instances = []
        _FakeEngine.fail_in = None
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def _config(self, *, train_entries=None, val_entries=None) -> Path:
        data_root = self.base / "data"
        (data_root / "train").mkdir(parents=True, exist_ok=True)
        checkpoint = self.base / "sam.pth"
        checkpoint.write_bytes(b"unused")
        train_entries = train_entries or [
            f"train/{index}_sat.jpg" for index in range(4)
        ]
        val_entries = val_entries or [
            f"train/v{index}_sat.jpg" for index in range(4)
        ]
        train_split = self.base / "train.json"
        train_split.write_text(json.dumps(train_entries), encoding="utf-8")
        val_split = self.base / "val.json"
        val_split.write_text(json.dumps(val_entries), encoding="utf-8")
        config = {
            "seed": 42,
            "data": {
                "root": str(data_root),
                "train_split": str(train_split),
                "val_split": str(val_split),
                "num_workers": 0,
            },
            "model": {"sam_checkpoint": str(checkpoint)},
            "stages": {
                "A": {"epochs": 1, "lr": 1e-4},
                "B": {"epochs": 1, "lr": 1e-4},
                "C": {"epochs": 1, "lr": 5e-5},
                "D": {"epochs": 1, "lr": 1e-6},
            },
            "rl": {"group_size": 4, "update_epochs": 1},
        }
        path = self.base / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def _run(self, config_path: Path, *extra, out_name="out"):
        out = self.base / out_name
        argv = ["--config", str(config_path), "--output-dir", str(out), *extra]
        with patch.object(probe, "DiffPeakEngine", _FakeEngine), patch.object(
            probe, "environment_info", return_value={"test": True}
        ):
            with self.assertRaises(SystemExit) as raised:
                probe.main(argv)
        return raised.exception.code, out


class BoundedIterableTest(unittest.TestCase):
    def test_cycles_up_to_bound_and_records_fetch_time(self):
        bounded = probe.BoundedIterable([1, 2, 3], 5, cycle=True)
        self.assertEqual(len(bounded), 5)
        self.assertEqual(list(bounded), [1, 2, 3, 1, 2])
        self.assertEqual(bounded.items_yielded, 5)
        self.assertGreaterEqual(bounded.fetch_seconds, 0.0)

    def test_stops_at_end_without_cycle(self):
        bounded = probe.BoundedIterable([1, 2, 3], 5, cycle=False)
        self.assertEqual(list(bounded), [1, 2, 3])
        self.assertEqual(bounded.items_yielded, 3)

    def test_empty_cycle_source_raises(self):
        with self.assertRaises(ValueError):
            list(probe.BoundedIterable([], 3, cycle=True))

    def test_sink_receives_fetch_records(self):
        records = []
        bounded = probe.BoundedIterable(
            [1, 2], 3, cycle=True, phase="measure", sink=records.append
        )
        self.assertEqual(list(bounded), [1, 2, 1])
        self.assertEqual([item["index"] for item in records], [1, 2, 3])
        self.assertEqual(records[0]["phase"], "measure")
        self.assertIn("fetch_seconds", records[0])

    def test_counts_actual_images_including_partial_batches(self):
        source = [
            {"image": torch.zeros(2, 1, 1, 1)},
            {"image": torch.zeros(1, 1, 1, 1)},
        ]
        bounded = probe.BoundedIterable(source, 2, cycle=False)
        produced = list(bounded)
        self.assertEqual(len(produced), 2)
        self.assertIs(produced[0], source[0])
        self.assertEqual(bounded.counted_samples(fallback_batch_size=2), 3)
        # Sources without image tensors keep the items * batch size estimate.
        fallback = probe.BoundedIterable([1, 2, 3], 3, cycle=False)
        list(fallback)
        self.assertEqual(fallback.counted_samples(fallback_batch_size=4), 12)


class SplitGuardrailTest(ProbeTestCase):
    def test_subset_is_deterministic_order_independent_and_bounded(self):
        entries = [f"train/{index}_sat.jpg" for index in range(20)]
        first = probe.select_train_subset(entries, 5, 7)
        second = probe.select_train_subset(list(reversed(entries)), 5, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(set(first).issubset(set(entries)))
        self.assertEqual(
            probe.select_train_subset(entries[:2], 99, 7), sorted(entries[:2])
        )

    def test_training_entries_reject_forbidden_directories(self):
        for entry in (
            "test/1_sat.jpg",
            "tests/1_sat.jpg",
            "valid/1_sat.jpg",
            "validation/1_sat.jpg",
            "val/1_sat.jpg",
            "TRAIN/VAL/1_sat.jpg",
            "/abs/1_sat.jpg",
            "train/../val/1_sat.jpg",
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    probe.assert_train_entries_allowed([entry])

    def test_subset_rejects_validation_overlap_by_path_and_id(self):
        train = ["train/1_sat.jpg", "train/2_sat.jpg"]
        with self.assertRaises(ValueError):
            probe.select_train_subset(train, 2, 42, val_entries=["train/2_sat.jpg"])
        with self.assertRaises(ValueError):
            probe.select_train_subset(
                train, 2, 42, val_entries=["other/2_sat.jpg"]
            )

    def test_subset_rejects_absolute_and_parent_escape_entries(self):
        # Normalization must not hide guardrail violations before the check.
        for entry in (
            "/abs/1_sat.jpg",
            "../train/1_sat.jpg",
            "./../train/1_sat.jpg",
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    probe.select_train_subset([entry], 1, 7)
        self.assertEqual(
            probe.select_train_subset(["./train/1_sat.jpg"], 1, 7),
            ["train/1_sat.jpg"],
        )

    def test_clean_disjoint_subset_passes(self):
        picked = probe.select_train_subset(
            ["train/1_sat.jpg", "train/2_sat.jpg"],
            2,
            42,
            val_entries=["train/3_sat.jpg"],
        )
        self.assertEqual(len(picked), 2)


class ConfigGuardrailTest(ProbeTestCase):
    def test_parser_defaults(self):
        args = probe.build_parser().parse_args(
            ["--output-dir", "out", "--stage", "A"]
        )
        self.assertEqual(args.config, "configs/3090.yaml")
        self.assertEqual(args.train_samples, 32)
        self.assertEqual(args.val_samples, 16)
        self.assertEqual(args.warmup_updates, 2)
        self.assertEqual(args.measure_updates, 4)
        self.assertIsNone(args.save_weights)

    def test_validate_args_rejects_invalid_bounds(self):
        base = [
            "--output-dir", "out", "--stage", "A",
            "--warmup-updates", "-1",
        ]
        args = probe.build_parser().parse_args(base)
        with self.assertRaises(ValueError):
            probe.validate_args(args)
        for flag, value in (
            ("--measure-updates", "0"),
            ("--train-samples", "0"),
            ("--val-samples", "0"),
            ("--batch-size", "0"),
        ):
            with self.subTest(flag=flag):
                args = probe.build_parser().parse_args(
                    [
                        "--output-dir", "out", "--stage", "A",
                        flag, value,
                    ]
                )
                with self.assertRaises(ValueError):
                    probe.validate_args(args)

    def test_overrides_are_in_memory_only(self):
        args = probe.build_parser().parse_args(
            [
                "--output-dir", "out",
                "--stage", "D",
                "--data-root", "/tmp/data",
                "--sam-checkpoint", "/tmp/sam.pth",
                "--batch-size", "2",
                "--accumulation-steps", "3",
                "--num-workers", "0",
                "--gradient-checkpointing", "off",
                "--lr", "1.5e-5",
                "--amp-init-scale", "32",
                "--group-size", "2",
                "--sampling-steps", "8",
                "--seed", "99",
                "--train-split", "custom_train.json",
                "--val-split", "custom_val.json",
            ]
        )
        original = copy.deepcopy(DEFAULT_CONFIG)
        resolved = probe.apply_overrides(DEFAULT_CONFIG, args)
        self.assertEqual(resolved["seed"], 99)
        self.assertEqual(resolved["data"]["root"], "/tmp/data")
        self.assertEqual(resolved["data"]["train_split"], "custom_train.json")
        self.assertEqual(resolved["data"]["val_split"], "custom_val.json")
        self.assertEqual(resolved["data"]["num_workers"], 0)
        self.assertEqual(resolved["model"]["sam_checkpoint"], "/tmp/sam.pth")
        self.assertFalse(resolved["model"]["gradient_checkpointing"])
        self.assertEqual(resolved["train"]["batch_size"], 2)
        self.assertEqual(resolved["train"]["accumulation_steps"], 3)
        self.assertEqual(resolved["stages"]["D"]["lr"], 1.5e-5)
        self.assertEqual(resolved["stages"]["A"]["lr"], 1e-4)
        self.assertEqual(resolved["rl"]["amp_init_scale"], 32.0)
        self.assertEqual(resolved["rl"]["group_size"], 2)
        self.assertEqual(resolved["diffusion"]["sampling_steps"], 8)
        self.assertEqual(DEFAULT_CONFIG, original)

    def test_supervised_amp_scale_goes_to_train(self):
        args = probe.build_parser().parse_args(
            [
                "--output-dir", "out", "--stage", "B",
                "--amp-init-scale", "16", "--gradient-checkpointing", "on",
            ]
        )
        resolved = probe.apply_overrides(DEFAULT_CONFIG, args)
        self.assertEqual(resolved["train"]["amp_init_scale"], 16.0)
        self.assertTrue(resolved["model"]["gradient_checkpointing"])

    def test_output_dir_must_be_unique(self):
        out = self.base / "out"
        out.mkdir()
        (out / "keep.txt").write_text("old", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            probe.prepare_output_dir(out)
        self.assertEqual((out / "keep.txt").read_text(encoding="utf-8"), "old")


class ProbeFlowTest(ProbeTestCase):
    def test_supervised_probe_bounds_phases_and_writes_strict_json(self):
        config_path = self._config()
        code, out = self._run(
            config_path,
            "--stage", "A",
            "--batch-size", "2",
            "--accumulation-steps", "2",
            "--warmup-updates", "2",
            "--measure-updates", "4",
        )
        self.assertEqual(code, 0)
        engine = _FakeEngine.instances[-1]
        self.assertEqual(engine.prepare_calls, [("A", 1, 6)])
        # 每个优化器组 = 2 个 micro-batch；warmup 2 组，measurement 4 组。
        self.assertEqual(
            engine.epoch_calls,
            [("supervised", "A", 0, 4), ("supervised", "A", 0, 8)],
        )
        results, text = _strict_json(out / "results.json")
        self.assertNotIn("NaN", text)
        self.assertEqual(results["status"], "ok")
        self.assertEqual(results["warmup"]["batches"], 4)
        self.assertEqual(results["measurement"]["batches"], 8)
        self.assertEqual(
            results["measurement"]["optimizer"]["attempted_updates"], 4
        )
        self.assertEqual(
            results["measurement"]["optimizer"]["successful_updates"], 4
        )
        self.assertGreaterEqual(results["measurement"]["dataloader_seconds"], 0.0)
        self.assertEqual(results["measurement"]["metrics"]["aux"], None)
        self.assertFalse(results["measurement"]["finite_metrics"]["aux"])
        self.assertFalse(results["quality_claim"])
        for name in (
            "environment.json", "threading.json", "resolved_config.json",
            "splits.json", "train_subset.json", "val_subset.json",
            "raw.jsonl", "results.json",
        ):
            self.assertTrue((out / name).is_file(), name)

        subset = json.loads((out / "train_subset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(subset), 4)
        splits = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(splits["train"]["source_count"], 4)
        self.assertEqual(splits["train"]["subset_count"], 4)
        self.assertEqual(splits["val"]["subset_used_for_training"], False)
        # The recorded file hash must match the written artifact.
        self.assertEqual(
            splits["train"]["subset_sha256"],
            probe.sha256_file(out / "train_subset.json"),
        )
        self.assertEqual(
            splits["train"]["subset_entries_sha256"], probe.sha256_entries(subset)
        )
        self.assertEqual(
            splits["val"]["subset_sha256"],
            probe.sha256_file(out / "val_subset.json"),
        )
        self.assertEqual(
            engine.config["data"]["train_split"],
            str((out / "train_subset.json").resolve()),
        )
        events = [
            json.loads(line)
            for line in (out / "raw.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        names = [item["event"] for item in events]
        for expected in (
            "probe_start", "splits_ready", "warmup_start", "warmup_end",
            "measurement_start", "measurement_end", "results_written", "probe_end",
        ):
            self.assertIn(expected, names)
        self.assertEqual(names.count("batch"), 12)

    def test_grpo_counts_groups_and_saves_model_only_weights(self):
        config_path = self._config()
        weights = self.base / "weights" / "bootstrap_D.pth"
        code, out = self._run(
            config_path,
            "--stage", "D",
            "--group-size", "2",
            "--init", str(self.base / "stage_C.pth"),
            "--warmup-updates", "2",
            "--measure-updates", "4",
            "--save-weights", str(weights),
        )
        self.assertEqual(code, 0)
        engine = _FakeEngine.instances[-1]
        self.assertEqual(engine.prepare_calls, [("D", 1, 6)])
        self.assertEqual(
            engine.epoch_calls, [("grpo", "D", 0, 2), ("grpo", "D", 0, 4)]
        )
        self.assertEqual(engine.init_calls, [str(self.base / "stage_C.pth")])
        results, _ = _strict_json(out / "results.json")
        optimizer = results["measurement"]["optimizer"]
        self.assertEqual(optimizer["attempted_groups"], 4)
        self.assertEqual(optimizer["successful_updates"], 4)
        self.assertEqual(optimizer["zero_update_groups"], 0.0)
        self.assertEqual(results["weights"]["path"], str(weights.resolve()))
        self.assertFalse(results["weights"]["resumable"])
        self.assertFalse(any("容量 smoke" in note for note in results["notes"]))
        payload = torch.load(weights, map_location="cpu", weights_only=False)
        self.assertEqual(payload["stage"], "D")
        self.assertEqual(payload["probe"]["kind"], "development_probe")
        self.assertIn("ref_unet", payload["models"])
        self.assertEqual(payload["format_version"], 3)
        self.assertIn("architecture", payload["sam"])
        self.assertIn("config", payload)

    def test_grpo_counts_groups_per_image_not_per_batch(self):
        config_path = self._config()
        code, out = self._run(
            config_path,
            "--stage", "D",
            "--batch-size", "2",
            "--group-size", "2",
            "--warmup-updates", "0",
            "--measure-updates", "4",
        )
        self.assertEqual(code, 0)
        engine = _FakeEngine.instances[-1]
        # 4 trajectory groups at batch 2 -> 2 batches; one step per batch.
        self.assertEqual(engine.prepare_calls, [("D", 1, 2)])
        self.assertEqual(engine.epoch_calls, [("grpo", "D", 0, 2)])
        results, _ = _strict_json(out / "results.json")
        optimizer = results["measurement"]["optimizer"]
        self.assertEqual(optimizer["attempted_batches"], 2)
        self.assertEqual(optimizer["attempted_groups"], 4)
        self.assertEqual(optimizer["successful_updates"], 2)
        self.assertEqual(results["measurement"]["samples"], 4)
        self.assertIn("groups_per_second", results["measurement"]["throughput"])

    def test_untrained_c_warns_capacity_smoke(self):
        config_path = self._config()
        code, out = self._run(
            config_path,
            "--stage", "C",
            "--warmup-updates", "0",
            "--measure-updates", "1",
        )
        self.assertEqual(code, 0)
        results, _ = _strict_json(out / "results.json")
        self.assertTrue(
            any("容量 smoke" in note for note in results["notes"])
        )
        self.assertFalse(results["quality_claim"])

    def test_validation_uses_only_fixed_subset(self):
        config_path = self._config()
        code, out = self._run(
            config_path,
            "--stage", "A",
            "--warmup-updates", "0",
            "--measure-updates", "1",
            "--validate",
            "--val-samples", "2",
        )
        self.assertEqual(code, 0)
        self.assertEqual(_FakeEngine.instances[-1].validate_calls, 1)
        validation, _ = _strict_json(out / "validation.json")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["subset_count"], 2)
        self.assertFalse(validation["quality_claim"])
        subset = json.loads((out / "val_subset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(subset), 2)

    def test_validation_reports_actual_subset_not_requested_count(self):
        config_path = self._config(
            val_entries=["train/v0_sat.jpg", "train/v1_sat.jpg"]
        )
        code, out = self._run(
            config_path,
            "--stage", "A",
            "--warmup-updates", "0",
            "--measure-updates", "1",
            "--validate",
            "--val-samples", "16",
        )
        self.assertEqual(code, 0)
        validation, _ = _strict_json(out / "validation.json")
        self.assertEqual(validation["subset_count"], 2)
        results, _ = _strict_json(out / "results.json")
        self.assertEqual(results["probe"]["val_samples"], 2)


class ProbeFailureTest(ProbeTestCase):
    def test_oom_is_recorded_and_prior_output_preserved(self):
        config_path = self._config()
        _FakeEngine.fail_in = "measure"
        code, out = self._run(
            config_path, "--stage", "A",
            "--warmup-updates", "1", "--measure-updates", "1",
        )
        self.assertEqual(code, 1)
        self.assertFalse((out / "results.json").exists())
        error, _ = _strict_json(out / "error.json")
        self.assertEqual(error["status"], "failed")
        self.assertEqual(error["phase"], "measure")
        self.assertTrue(error["out_of_memory"])
        self.assertIn("OutOfMemoryError", error["error_type"])
        self.assertIn("Traceback", error["traceback"])
        # 失败前写出的证据仍在。
        self.assertTrue((out / "resolved_config.json").exists())
        self.assertTrue((out / "splits.json").exists())
        self.assertTrue((out / "environment.json").exists())
        events = [
            json.loads(line)
            for line in (out / "raw.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(any(item["event"] == "probe_end" and item["status"] == "failed"
                            for item in events))

    def test_validation_failure_keeps_results(self):
        config_path = self._config()
        _FakeEngine.fail_in = "validate"
        code, out = self._run(
            config_path, "--stage", "A",
            "--warmup-updates", "0", "--measure-updates", "1", "--validate",
        )
        self.assertEqual(code, 1)
        self.assertTrue((out / "results.json").exists())
        self.assertTrue((out / "validation_error.json").exists())
        self.assertFalse((out / "error.json").exists())
        error, _ = _strict_json(out / "validation_error.json")
        self.assertEqual(error["error_type"], "RuntimeError")
        self.assertIn("simulated validation failure", error["message"])

    def test_save_weights_refuses_to_overwrite_existing_file(self):
        config_path = self._config()
        target = self.base / "existing.pth"
        target.write_bytes(b"keep")
        code, out = self._run(
            config_path, "--stage", "A",
            "--warmup-updates", "0", "--measure-updates", "1",
            "--save-weights", str(target),
        )
        self.assertEqual(code, 1)
        self.assertEqual(target.read_bytes(), b"keep")
        error, _ = _strict_json(out / "error.json")
        self.assertEqual(error["phase"], "weights")
        # Evidence written before the save step stays intact.
        self.assertTrue((out / "resolved_config.json").exists())
        self.assertTrue((out / "splits.json").exists())
        self.assertFalse((out / "results.json").exists())


class FiniteScopeRegressionTest(unittest.TestCase):
    """Absent stage outputs must not make all_metrics_finite false."""

    def _engine(self, kl_weight=0.0, anchor_weight=0.0):
        engine = object.__new__(_FakeEngine)
        engine.kl_weight = kl_weight
        engine.anchor_weight = anchor_weight
        return engine

    def _loader(self):
        return probe.BoundedIterable([], 0)

    def _summarize(self, stage, stats, engine):
        keys = probe._applicable_metric_keys(stage, engine, stats)
        return probe._phase_summary(
            stage, stats, 1.0, self._loader(), 2, 1, None, {},
            applicable_metrics=keys,
        )

    def test_stage_a_aux_diff_heat_absent_but_loss_finite(self):
        stats = {
            "loss": 0.727, "seg": 0.5, "points": 12.0,
            "aux": float("nan"), "diff": float("nan"), "heat": float("nan"),
            "optimizer_steps": 16.0, "skipped_groups": 0.0,
        }
        summary = self._summarize("A", stats, self._engine())
        self.assertTrue(summary["all_metrics_finite"])
        self.assertTrue(summary["core_metrics_finite"])
        self.assertTrue(summary["finite_metrics"]["loss"])
        self.assertFalse(summary["finite_metrics"]["aux"])
        self.assertIn("aux", summary["intentionally_absent_metrics"])
        self.assertEqual(summary["applicable_metrics"], ["loss", "seg", "points"])

    def test_stage_a_nonfinite_loss_still_reports_false(self):
        stats = {
            "loss": float("nan"), "seg": 0.5, "points": 12.0,
            "aux": float("nan"), "diff": float("nan"), "heat": float("nan"),
            "optimizer_steps": 1.0, "skipped_groups": 0.0,
        }
        summary = self._summarize("A", stats, self._engine())
        self.assertFalse(summary["all_metrics_finite"])

    def test_stage_b_checks_all_computed_losses(self):
        stats = {
            "loss": 2.0, "diff": 1.0, "heat": 0.2, "aux": 0.3,
            "seg": float("nan"), "points": float("nan"),
            "optimizer_steps": 1.0, "skipped_groups": 0.0,
        }
        summary = self._summarize("B", stats, self._engine())
        self.assertTrue(summary["all_metrics_finite"])
        self.assertIn("seg", summary["intentionally_absent_metrics"])

    def test_stage_d_disabled_kl_anchor_are_not_required(self):
        stats = {
            "reward": 0.5, "iou": 0.2, "grpo": -0.1,
            "kl": float("nan"), "anchor": float("nan"), "policy_terms": 1.0,
            "optimizer_steps": 1.0, "skipped_update_steps": 0.0,
        }
        keys = probe._applicable_metric_keys("D", self._engine(), stats)
        summary = probe._phase_summary(
            "D", stats, 1.0, self._loader(), 1, 1, None, {},
            applicable_metrics=keys,
        )
        self.assertTrue(summary["all_metrics_finite"])
        self.assertNotIn("kl", keys)
        self.assertNotIn("anchor", keys)

    def test_stage_d_grpo_nan_with_policy_terms_fails(self):
        stats = {
            "reward": 0.5, "iou": 0.2, "grpo": float("nan"),
            "kl": float("nan"), "anchor": float("nan"), "policy_terms": 3.0,
            "optimizer_steps": 1.0,
        }
        summary = self._summarize("D", stats, self._engine())
        self.assertFalse(summary["all_metrics_finite"])


class StrictJsonTest(unittest.TestCase):
    def test_multi_element_tensor_is_stringified_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            probe.write_json(path, {"tensor": torch.tensor([1.0, 2.0])})
            data, text = _strict_json(path)
            self.assertIsInstance(data["tensor"], str)
            self.assertNotIn("NaN", text)


if __name__ == "__main__":
    unittest.main()
