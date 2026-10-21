"""scripts/gpu_telemetry.py 与 probe 阶段集成的 mock 测试。

模块部分只用标准库（+ unittest.mock），不依赖 torch/CUDA/nvidia-smi；集成
部分用假引擎在 CPU 上验证 warmup/measurement 使用独立文件、finally 关闭、
以及 measurement 不会混入 initialization/validation 采样。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_SCRIPT = ROOT / "scripts" / "gpu_telemetry.py"
PROBE_SCRIPT = ROOT / "scripts" / "probe_3090.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


telemetry = _load("gpu_telemetry_under_test", TELEMETRY_SCRIPT)

ROWS = [
    telemetry.GpuRow(0, "GPU-aaaa", "NVIDIA RTX 3090"),
    telemetry.GpuRow(1, "GPU-bbbb", "NVIDIA RTX 3090"),
    telemetry.GpuRow(2, "GPU-cccc", "NVIDIA RTX 3090"),
]
DISCOVERY = "\n".join(
    f"{row.index}, {row.uuid}, {row.name}" for row in ROWS
)


def _wait_until(predicate, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class _FakeSmi:
    """Discovery + per-GPU sample fixture with thread-safe call capture."""

    def __init__(
        self,
        *,
        discovery: str = DISCOVERY,
        samples=None,
        fail_samples: bool = False,
    ) -> None:
        self.discovery = discovery
        self.samples = samples or {
            "GPU-aaaa": "0, GPU-aaaa, 10, 100, 20.0, 40",
            "GPU-bbbb": "1, GPU-bbbb, 80, 2048, 250.5, 65",
            "GPU-cccc": "2, GPU-cccc, 55, 512, 100.25, 50",
        }
        self.fail_samples = fail_samples
        self.sample_calls = []
        self._lock = threading.Lock()

    @staticmethod
    def _is_discovery(args) -> bool:
        return bool(args) and str(args[0]).startswith(
            "--query-gpu=index,uuid,name"
        )

    def __call__(self, args):
        args = list(args)
        if self._is_discovery(args):
            return self.discovery
        with self._lock:
            self.sample_calls.append(args)
        if self.fail_samples:
            raise RuntimeError("simulated nvidia-smi failure")
        selector = str(args[0]).split("=", 1)[1]
        if selector not in self.samples:
            raise RuntimeError(f"unexpected selector {selector}")
        return self.samples[selector]


class ParseAndMappingTest(unittest.TestCase):
    def test_parse_gpu_rows_keeps_name_commas_and_skips_garbage(self):
        text = (
            "0, GPU-aaaa, RTX 3090, NVLink\n"
            "not-an-index, GPU-x, GPU\n"
            "\n"
            "1, GPU-bbbb\n"
        )
        rows = telemetry.parse_gpu_rows(text)
        self.assertEqual([row.index for row in rows], [0, 1])
        self.assertEqual(rows[0].name, "RTX 3090, NVLink")
        self.assertEqual(rows[1].name, "")

    def test_parse_sample_line_maps_unavailable_values_to_none(self):
        sample = telemetry.parse_sample_line(
            "1, GPU-bbbb, [N/A], 2048, N/A, 65.0\n"
        )
        self.assertEqual(sample["index"], 1)
        self.assertEqual(sample["uuid"], "GPU-bbbb")
        self.assertIsNone(sample["utilization_gpu_percent"])
        self.assertEqual(sample["memory_used_mib"], 2048.0)
        self.assertIsNone(sample["power_draw_w"])
        self.assertEqual(sample["temperature_c"], 65.0)
        self.assertIsNone(telemetry.parse_sample_line("[N/A]"))

    def test_summarize_samples_ignores_missing_and_computes_stats(self):
        records = [
            {"utilization_gpu_percent": 10.0, "memory_used_mib": 100},
            {"utilization_gpu_percent": 30.0, "memory_used_mib": None},
            {"utilization_gpu_percent": None, "memory_used_mib": 300},
        ]
        summary = telemetry.summarize_samples(records)
        utilization = summary["utilization_gpu_percent"]
        self.assertEqual(utilization["count"], 2)
        self.assertEqual(utilization["min"], 10.0)
        self.assertEqual(utilization["max"], 30.0)
        self.assertEqual(utilization["mean"], 20.0)
        self.assertEqual(utilization["last"], 30.0)
        memory = summary["memory_used_mib"]
        self.assertEqual(memory["count"], 2)
        self.assertEqual(memory["mean"], 200.0)
        self.assertEqual(summary["power_draw_w"]["count"], 0)
        self.assertIsNone(summary["power_draw_w"]["min"])

    def test_unset_visibility_single_gpu_is_safe(self):
        single = [telemetry.GpuRow(0, "GPU-aaaa", "NVIDIA RTX 3090")]
        resolution = telemetry.resolve_target_device("cuda:0", None, single)
        self.assertIsNone(resolution.reason)
        self.assertEqual(resolution.target.physical_index, 0)
        self.assertEqual(resolution.target.selector, "single_gpu_default")
        out_of_range = telemetry.resolve_target_device("cuda:1", None, single)
        self.assertIsNone(out_of_range.target)
        self.assertEqual(out_of_range.reason, "logical_device_out_of_range")

    def test_unset_visibility_multi_gpu_is_ambiguous_not_guessed(self):
        resolution = telemetry.resolve_target_device("cuda:0", None, ROWS)
        self.assertIsNone(resolution.target)
        self.assertEqual(resolution.reason, "ambiguous_enumeration_order")

    def test_visible_devices_remap_logical_index(self):
        resolution = telemetry.resolve_target_device("cuda:1", "2,0", ROWS)
        self.assertIsNone(resolution.reason)
        self.assertEqual(resolution.target.physical_index, 0)
        self.assertEqual(resolution.target.uuid, "GPU-aaaa")
        self.assertEqual(
            resolution.target.selector, "cuda_visible_devices_index"
        )

    def test_visible_device_uuid_and_prefix_routing(self):
        exact = telemetry.resolve_target_device("cuda:0", "GPU-bbbb", ROWS)
        self.assertEqual(exact.target.physical_index, 1)
        self.assertEqual(exact.target.selector, "cuda_visible_devices_uuid")
        prefixed = telemetry.resolve_target_device("cuda:0", "GPU-cc", ROWS)
        self.assertEqual(prefixed.target.physical_index, 2)

    def test_ambiguous_prefix_is_rejected(self):
        resolution = telemetry.resolve_target_device("cuda:0", "GPU-", ROWS)
        self.assertIsNone(resolution.target)
        self.assertEqual(resolution.reason, "ambiguous_uuid")

    def test_unmappable_and_invalid_visibility_are_reported(self):
        cases = [
            ("MIG-1234", "unmapped_visible_device"),
            ("", "no_visible_devices"),
            ("-1", "no_visible_devices"),
            ("0,,1", "invalid_cuda_visible_devices"),
            ("0,1", "logical_device_out_of_range"),
            ("9", "physical_index_not_found"),
        ]
        for visible, expected in cases:
            with self.subTest(visible=visible):
                device = "cuda:2" if visible == "0,1" else "cuda:0"
                resolution = telemetry.resolve_target_device(
                    device, visible, ROWS
                )
                self.assertIsNone(resolution.target)
                self.assertEqual(resolution.reason, expected)

    def test_device_uuid_is_authoritative_over_enumeration_order(self):
        # Logical cuda:0 is actually GPU-c under torch, regardless of
        # nvidia-smi index order or CUDA_DEVICE_ORDER=FASTEST_FIRST.
        resolution = telemetry.resolve_target_device(
            "cuda:0", None, ROWS, device_uuid="cccc"
        )
        self.assertIsNone(resolution.reason)
        self.assertEqual(resolution.target.physical_index, 2)
        self.assertEqual(resolution.target.uuid, "GPU-cccc")
        self.assertEqual(resolution.target.selector, "torch_device_uuid")

        prefixed = telemetry.resolve_target_device(
            "cuda:0", "MIG-ignored", ROWS, device_uuid="GPU-bbbb"
        )
        self.assertEqual(prefixed.target.physical_index, 1)

    def test_device_uuid_unmappable_is_reported(self):
        mig = telemetry.resolve_target_device(
            "cuda:0", "0", ROWS, device_uuid="MIG-0123"
        )
        self.assertEqual(mig.reason, "unmapped_visible_device")
        missing = telemetry.resolve_target_device(
            "cuda:0", "0", ROWS, device_uuid="GPU-zzzz"
        )
        self.assertEqual(missing.reason, "uuid_not_found")

    def test_cpu_and_empty_smi_are_unavailable(self):
        cpu = telemetry.resolve_target_device("cpu", "0", ROWS)
        self.assertEqual(cpu.reason, "not_cuda")
        empty = telemetry.resolve_target_device("cuda", None, [])
        self.assertEqual(empty.reason, "nvidia_smi_unavailable")
        invalid = telemetry.resolve_target_device("cuda:abc", None, ROWS)
        self.assertEqual(invalid.reason, "invalid_device")


class SamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_samples_only_selected_gpu_and_writes_timestamped_jsonl(self):
        fake = _FakeSmi()
        path = self.base / "gpu_telemetry_measurement.jsonl"
        sampler = telemetry.GpuTelemetrySampler(
            path,
            device="cuda:1",
            cuda_visible_devices="0,1",
            interval=0.01,
            runner=fake,
        )
        sampler.start()
        self.assertTrue(
            _wait_until(lambda: sampler.summary()["sample_count"] >= 2)
        )
        summary = sampler.stop()

        self.assertTrue(summary["available"])
        self.assertIsNone(summary["reason"])
        self.assertEqual(summary["device"]["physical_index"], 1)
        self.assertEqual(summary["device"]["uuid"], "GPU-bbbb")
        self.assertEqual(
            summary["device"]["selector"], "cuda_visible_devices_index"
        )
        self.assertEqual(summary["raw_jsonl"], path.name)
        self.assertGreaterEqual(summary["sample_count"], 2)
        self.assertEqual(summary["error_count"], 0)
        # Every metric query names exactly the selected physical GPU.
        for args in fake.sample_calls:
            self.assertEqual(args[0], "--id=GPU-bbbb")
            self.assertNotIn("GPU-aaaa", " ".join(args))
        self.assertEqual(
            summary["metrics"]["utilization_gpu_percent"]["min"], 80.0
        )
        self.assertEqual(
            summary["metrics"]["memory_used_mib"]["last"], 2048.0
        )

        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(lines), 2)
        for record in lines:
            self.assertIn("time", record)
            self.assertIn("time_utc", record)
            self.assertGreaterEqual(record["elapsed_seconds"], 0.0)
            self.assertEqual(record["uuid"], "GPU-bbbb")
            self.assertEqual(record["utilization_gpu_percent"], 80.0)

        # The sampling thread is gone after stop.
        time.sleep(0.03)
        self.assertEqual(sampler.summary()["sample_count"], summary["sample_count"])

    def test_uuid_visibility_selects_matching_physical_gpu(self):
        fake = _FakeSmi()
        sampler = telemetry.GpuTelemetrySampler(
            self.base / "uuid.jsonl",
            device="cuda:0",
            cuda_visible_devices="GPU-cccc",
            interval=0.01,
            runner=fake,
        )
        sampler.start()
        self.assertTrue(
            _wait_until(lambda: sampler.summary()["sample_count"] >= 1)
        )
        summary = sampler.stop()
        self.assertEqual(summary["device"]["physical_index"], 2)
        self.assertTrue(fake.sample_calls)
        self.assertTrue(
            all(args[0] == "--id=GPU-cccc" for args in fake.sample_calls)
        )

    def test_torch_device_uuid_overrides_unmappable_visibility(self):
        fake = _FakeSmi()
        sampler = telemetry.GpuTelemetrySampler(
            self.base / "torch_uuid.jsonl",
            device="cuda:0",
            cuda_visible_devices="MIG-0123",
            device_uuid="bbbb",
            interval=0.01,
            runner=fake,
        )
        sampler.start()
        self.assertTrue(
            _wait_until(lambda: sampler.summary()["sample_count"] >= 1)
        )
        summary = sampler.stop()
        self.assertTrue(summary["available"])
        self.assertEqual(summary["device"]["physical_index"], 1)
        self.assertEqual(summary["device"]["selector"], "torch_device_uuid")
        self.assertTrue(
            all(args[0] == "--id=GPU-bbbb" for args in fake.sample_calls)
        )

    def test_two_phases_write_independent_files_without_cross_mixing(self):
        fake = _FakeSmi()
        warmup_path = self.base / "gpu_telemetry_warmup.jsonl"
        measure_path = self.base / "gpu_telemetry_measurement.jsonl"
        warmup = telemetry.GpuTelemetrySampler(
            warmup_path, device="cuda:0", cuda_visible_devices="0",
            interval=0.01, runner=fake,
        )
        measurement = telemetry.GpuTelemetrySampler(
            measure_path, device="cuda:0", cuda_visible_devices="1",
            interval=0.01, runner=fake,
        )
        warmup.start()
        measurement.start()
        self.assertTrue(
            _wait_until(
                lambda: warmup.summary()["sample_count"] >= 1
                and measurement.summary()["sample_count"] >= 1
            )
        )
        warmup_summary = warmup.stop()
        measure_summary = measurement.stop()

        self.assertNotEqual(
            warmup_summary["raw_jsonl"], measure_summary["raw_jsonl"]
        )
        warmup_lines = [
            json.loads(line)
            for line in warmup_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measure_lines = [
            json.loads(line)
            for line in measure_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(warmup_lines)
        self.assertTrue(measure_lines)
        self.assertTrue(all(row["uuid"] == "GPU-aaaa" for row in warmup_lines))
        self.assertTrue(all(row["uuid"] == "GPU-bbbb" for row in measure_lines))

    def test_unset_visibility_multi_gpu_starts_no_sampling(self):
        fake = _FakeSmi()
        path = self.base / "ambiguous.jsonl"
        sampler = telemetry.GpuTelemetrySampler(
            path, device="cuda:0", cuda_visible_devices=None,
            interval=0.01, runner=fake,
        )
        sampler.start()
        summary = sampler.stop()
        self.assertFalse(summary["available"])
        self.assertEqual(summary["reason"], "ambiguous_enumeration_order")
        self.assertEqual(fake.sample_calls, [])
        self.assertFalse(path.exists())

    def test_unset_visibility_single_gpu_samples_the_only_gpu(self):
        fake = _FakeSmi(discovery="0, GPU-aaaa, NVIDIA RTX 3090")
        sampler = telemetry.GpuTelemetrySampler(
            self.base / "single.jsonl", device="cuda:0",
            cuda_visible_devices=None, interval=0.01, runner=fake,
        )
        sampler.start()
        self.assertTrue(
            _wait_until(lambda: sampler.summary()["sample_count"] >= 1)
        )
        summary = sampler.stop()
        self.assertTrue(summary["available"])
        self.assertEqual(summary["device"]["selector"], "single_gpu_default")
        self.assertTrue(
            all(args[0] == "--id=GPU-aaaa" for args in fake.sample_calls)
        )

    def test_sampling_failure_is_recorded_and_never_raised(self):
        fake = _FakeSmi(fail_samples=True)
        path = self.base / "failing.jsonl"
        sampler = telemetry.GpuTelemetrySampler(
            path, device="cuda:0", cuda_visible_devices="0",
            interval=0.005, runner=fake,
        )
        sampler.start()
        self.assertTrue(
            _wait_until(lambda: sampler.summary()["error_count"] >= 1)
        )
        summary = sampler.stop()
        self.assertTrue(summary["available"])
        self.assertEqual(summary["sample_count"], 0)
        self.assertGreaterEqual(summary["error_count"], 1)
        self.assertIn("RuntimeError", summary["last_error"])
        self.assertTrue(path.is_file())

    def test_mismatched_sample_uuid_is_rejected(self):
        fake = _FakeSmi(
            samples={"GPU-bbbb": "0, GPU-aaaa, 99, 4096, 300, 90"}
        )
        path = self.base / "mismatch.jsonl"
        sampler = telemetry.GpuTelemetrySampler(
            path, device="cuda:1", cuda_visible_devices="0,1",
            interval=0.005, runner=fake,
        )
        sampler.start()
        self.assertTrue(
            _wait_until(lambda: sampler.summary()["error_count"] >= 1)
        )
        summary = sampler.stop()
        self.assertEqual(summary["sample_count"], 0)
        self.assertIn("非目标 GPU", summary["last_error"])
        self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_discovery_failure_reports_unavailable_without_file(self):
        def runner(args):
            raise FileNotFoundError("nvidia-smi not found")

        path = self.base / "missing.jsonl"
        sampler = telemetry.GpuTelemetrySampler(
            path, device="cuda", interval=0.01, runner=runner
        )
        sampler.start()
        summary = sampler.stop()
        self.assertFalse(summary["available"])
        self.assertIn("nvidia_smi_unavailable", summary["reason"])
        self.assertEqual(summary["sample_count"], 0)
        self.assertIsNone(summary["raw_jsonl"])
        self.assertFalse(path.exists())

    def test_mig_visibility_is_unavailable_and_not_queried(self):
        fake = _FakeSmi()
        sampler = telemetry.GpuTelemetrySampler(
            self.base / "mig.jsonl",
            device="cuda:0",
            cuda_visible_devices="MIG-0123",
            interval=0.01,
            runner=fake,
        )
        sampler.start()
        summary = sampler.stop()
        self.assertFalse(summary["available"])
        self.assertEqual(summary["reason"], "unmapped_visible_device")
        self.assertIsNone(summary["device"])
        self.assertEqual(fake.sample_calls, [])

    def test_stop_without_start_is_safe(self):
        sampler = telemetry.GpuTelemetrySampler(
            self.base / "never.jsonl", device="cuda", runner=_FakeSmi()
        )
        summary = sampler.stop()
        self.assertFalse(summary["available"])
        self.assertEqual(summary["reason"], "not_started")
        self.assertEqual(summary["sample_count"], 0)

    def test_unavailable_summary_schema_is_complete(self):
        summary = telemetry.unavailable_summary("not_cuda", interval_seconds=0.25)
        for key in (
            "available", "reason", "device", "interval_seconds",
            "sample_count", "error_count", "last_error", "raw_jsonl", "metrics",
        ):
            self.assertIn(key, summary)
        self.assertEqual(summary["metrics"]["utilization_gpu_percent"]["count"], 0)


class _FakeEngine:
    def __init__(self, device):
        self.device = device
        self.accumulation_steps = 1
        self.train_cfg = {"batch_size": 1}
        self.fail_training = False

    def _build_train_loader(self, epoch):
        return [{"index": 0}, {"index": 1}]

    def _train_epoch_supervised(self, stage, epoch, loader):
        for _ in loader:
            pass
        if self.fail_training:
            raise RuntimeError("simulated training failure")
        return {
            "loss": 1.0,
            "seg": 0.5,
            "optimizer_steps": 1.0,
            "skipped_groups": 0.0,
        }


class _RecordingSampler:
    instances = []

    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self.kwargs = kwargs
        self.started = False
        self.stop_calls = 0
        _RecordingSampler.instances.append(self)

    def start(self):
        self.started = True

    def summary(self):
        return {
            "available": True,
            "reason": None,
            "raw_jsonl": self.path.name,
            "sample_count": 0,
            "metrics": {},
        }

    def stop(self):
        self.stop_calls += 1
        return {
            "available": True,
            "reason": None,
            "raw_jsonl": self.path.name,
            "sample_count": 1,
            "metrics": {"utilization_gpu_percent": {"last": 50.0}},
        }


class ProbePhaseIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingSampler.instances = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        self._probe = _load("probe_3090_telemetry_under_test", PROBE_SCRIPT)

    def _cuda_patches(self):
        return [
            patch.object(
                self._probe, "_selected_cuda_uuid", return_value="GPU-test"
            ),
            patch.object(self._probe.torch.cuda, "synchronize", return_value=None),
            patch.object(
                self._probe.torch.cuda, "reset_peak_memory_stats",
                return_value=None,
            ),
            patch.object(
                self._probe.torch.cuda, "max_memory_allocated", return_value=0
            ),
            patch.object(
                self._probe.torch.cuda, "max_memory_reserved", return_value=0
            ),
        ]

    def test_measurement_and_warmup_use_distinct_telemetry_files(self):
        engine = _FakeEngine(self._probe.torch.device("cuda"))
        raw = self._probe.RawLog(self.out / "raw.jsonl")
        self.addCleanup(raw.close)
        with patch.object(
            self._probe, "GpuTelemetrySampler", _RecordingSampler
        ):
            patches = self._cuda_patches()
            for item in patches:
                item.start()
            try:
                warmup = self._probe.run_phase(
                    engine, "A", 1, "warmup", raw, output_dir=self.out
                )
                measurement = self._probe.run_phase(
                    engine, "A", 1, "measurement", raw, output_dir=self.out
                )
            finally:
                for item in patches:
                    item.stop()

        self.assertEqual(len(_RecordingSampler.instances), 2)
        self.assertEqual(
            _RecordingSampler.instances[0].path.name,
            "gpu_telemetry_warmup.jsonl",
        )
        self.assertEqual(
            _RecordingSampler.instances[1].path.name,
            "gpu_telemetry_measurement.jsonl",
        )
        self.assertNotEqual(
            _RecordingSampler.instances[0].path,
            _RecordingSampler.instances[1].path,
        )
        self.assertTrue(
            all(sampler.started for sampler in _RecordingSampler.instances)
        )
        self.assertTrue(
            all(sampler.stop_calls == 1 for sampler in _RecordingSampler.instances)
        )
        self.assertTrue(
            all(
                sampler.kwargs.get("device_uuid") == "GPU-test"
                for sampler in _RecordingSampler.instances
            )
        )
        self.assertEqual(
            warmup["gpu_telemetry"]["raw_jsonl"], "gpu_telemetry_warmup.jsonl"
        )
        self.assertEqual(
            measurement["gpu_telemetry"]["raw_jsonl"],
            "gpu_telemetry_measurement.jsonl",
        )

    def test_sampler_is_stopped_in_finally_when_training_raises(self):
        engine = _FakeEngine(self._probe.torch.device("cuda"))
        engine.fail_training = True
        raw = self._probe.RawLog(self.out / "raw.jsonl")
        self.addCleanup(raw.close)
        with patch.object(
            self._probe, "GpuTelemetrySampler", _RecordingSampler
        ):
            patches = self._cuda_patches()
            for item in patches:
                item.start()
            try:
                with self.assertRaises(RuntimeError):
                    self._probe.run_phase(
                        engine, "A", 1, "measurement", raw,
                        output_dir=self.out,
                    )
            finally:
                for item in patches:
                    item.stop()
        self.assertEqual(len(_RecordingSampler.instances), 1)
        self.assertEqual(_RecordingSampler.instances[0].stop_calls, 1)

    def test_selected_cuda_uuid_normalizes_and_degrades_to_none(self):
        with patch.object(
            self._probe.torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(uuid="abcd"),
        ):
            self.assertEqual(
                self._probe._selected_cuda_uuid(
                    self._probe.torch.device("cuda:0")
                ),
                "GPU-abcd",
            )
        with patch.object(
            self._probe.torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(uuid="MIG-1234"),
        ):
            self.assertEqual(
                self._probe._selected_cuda_uuid(
                    self._probe.torch.device("cuda:0")
                ),
                "MIG-1234",
            )
        with patch.object(
            self._probe.torch.cuda,
            "get_device_properties",
            side_effect=RuntimeError("no CUDA"),
        ):
            self.assertIsNone(
                self._probe._selected_cuda_uuid(
                    self._probe.torch.device("cuda:0")
                )
            )

    def test_cpu_device_records_unavailable_without_starting_sampler(self):
        engine = _FakeEngine(self._probe.torch.device("cpu"))
        raw = self._probe.RawLog(self.out / "raw.jsonl")
        self.addCleanup(raw.close)
        with patch.object(
            self._probe, "GpuTelemetrySampler",
            side_effect=AssertionError("CPU 不应启动 GPU 采样"),
        ):
            summary = self._probe.run_phase(
                engine, "A", 1, "measurement", raw, output_dir=self.out
            )
        gpu = summary["gpu_telemetry"]
        self.assertFalse(gpu["available"])
        self.assertEqual(gpu["reason"], "not_cuda")
        self.assertIsNone(gpu["raw_jsonl"])

    def test_empty_warmup_budget_records_unavailable_without_sampling(self):
        engine = _FakeEngine(self._probe.torch.device("cuda"))
        raw = self._probe.RawLog(self.out / "raw.jsonl")
        self.addCleanup(raw.close)
        with patch.object(
            self._probe, "GpuTelemetrySampler",
            side_effect=AssertionError("空阶段不应启动 GPU 采样"),
        ):
            summary = self._probe.run_phase(
                engine, "A", 0, "warmup", raw, output_dir=self.out
            )
        self.assertFalse(summary["gpu_telemetry"]["available"])
        self.assertEqual(summary["gpu_telemetry"]["reason"], "empty_phase")


if __name__ == "__main__":
    unittest.main()
