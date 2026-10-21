"""ExperimentLogger 的行为：JSONL、TensorBoard、tee、状态、会话与 RNG 隔离。"""
from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from diffpeak.experiment_logging import ExperimentLogger


def _config(**logging):
    settings = {"tensorboard": True, "log_every": 10, "flush_secs": 30}
    settings.update(logging)
    return {"logging": settings, "seed": 7}


def _metrics_records(directory: Path):
    path = directory / "metrics.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scalar_tags(directory: Path):
    accumulator = EventAccumulator(str(directory / "tensorboard"))
    accumulator.Reload()
    return accumulator


class ExperimentLoggerTest(unittest.TestCase):
    def test_session_files_events_and_completed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(tensorboard=False)
            with ExperimentLogger(root, config, metadata={"requested": "A"}) as logger:
                self.assertTrue(logger.directory.is_dir())
                self.assertEqual(
                    logger.directory.parent, root / "experiments"
                )
                logger.event("custom", detail="value")
                logger.log_scalars(
                    "A", "train_batch", {"loss": 1.25, "lr": 1e-4}, step=1, epoch=1
                )
                logger.flush()

            self.assertTrue((logger.directory / "config.json").is_file())
            self.assertTrue((logger.directory / "metadata.json").is_file())
            self.assertTrue((logger.directory / "train.log").is_file())
            metadata = json.loads(
                (logger.directory / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["session_id"], logger.session_id)
            self.assertEqual(metadata["requested"], "A")
            self.assertIn("started_at", metadata)
            self.assertIn("finished_at", metadata)
            self.assertGreaterEqual(metadata["duration_seconds"], 0.0)
            saved_config = json.loads(
                (logger.directory / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config, config)

            records = _metrics_records(logger.directory)
            self.assertEqual(records[0]["event"], "session_start")
            self.assertIn("custom", [record["event"] for record in records])
            metric = next(
                record for record in records if record["event"] == "metrics"
            )
            self.assertEqual(metric["stage"], "A")
            self.assertEqual(metric["scope"], "train_batch")
            self.assertEqual(metric["step"], 1)
            self.assertEqual(metric["epoch"], 1)
            self.assertEqual(metric["metrics"], {"loss": 1.25, "lr": 1e-4})
            self.assertEqual(records[-1]["event"], "session_end")
            self.assertEqual(records[-1]["status"], "completed")

    def test_tensorboard_scalars_are_readable_with_stage_scoped_tags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ExperimentLogger(root, _config()) as logger:
                logger.log_scalars(
                    "A", "train_batch", {"loss": 1.0}, step=1, epoch=1
                )
                logger.log_scalars(
                    "A", "train_batch", {"loss": 2.0}, step=2, epoch=1
                )
                logger.log_scalars(
                    "D", "train_group", {"reward": 0.5}, step=3, epoch=1
                )
                logger.log_scalars(
                    "A", "train_epoch", {"loss": 1.5}, step=1, epoch=1
                )
                logger.flush()
                directory = logger.directory

            accumulator = _scalar_tags(directory)
            self.assertEqual(
                sorted(accumulator.Tags()["scalars"]),
                [
                    "A/train_batch/loss",
                    "A/train_epoch/loss",
                    "D/train_group/reward",
                ],
            )
            points = [
                (event.step, event.value)
                for event in accumulator.Scalars("A/train_batch/loss")
            ]
            self.assertEqual(points, [(1, 1.0), (2, 2.0)])
            reward = accumulator.Scalars("D/train_group/reward")
            self.assertEqual([(event.step, event.value) for event in reward], [(3, 0.5)])

    def test_non_finite_scalars_are_json_null_and_not_written_to_tensorboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ExperimentLogger(root, _config()) as logger:
                logger.log_scalars(
                    "B",
                    "train_batch",
                    {
                        "loss": float("nan"),
                        "accuracy": float("inf"),
                        "negative": float("-inf"),
                        "finite": 0.25,
                    },
                    step=4,
                    epoch=2,
                )
                logger.flush()
                directory = logger.directory

            text = (directory / "metrics.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            metric = next(
                record
                for record in _metrics_records(directory)
                if record["event"] == "metrics"
            )
            self.assertEqual(
                metric["metrics"],
                {
                    "loss": None,
                    "accuracy": None,
                    "negative": None,
                    "finite": 0.25,
                },
            )
            self.assertTrue(
                all(
                    json.loads(line)
                    for line in text.splitlines()
                    if line.strip()
                )
            )

            accumulator = _scalar_tags(directory)
            self.assertEqual(accumulator.Tags()["scalars"], ["B/train_batch/finite"])
            points = accumulator.Scalars("B/train_batch/finite")
            self.assertEqual([(event.step, event.value) for event in points], [(4, 0.25)])

    def test_stdout_stderr_are_tailed_and_exception_marks_failed_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with ExperimentLogger(root, _config(tensorboard=False)) as logger:
                    print("progress on stdout")
                    print("warning on stderr", file=sys.stderr)
                    directory = logger.directory
                    raise RuntimeError("boom")

            self.assertIs(sys.stdout, original_stdout)
            self.assertIs(sys.stderr, original_stderr)
            log_text = (directory / "train.log").read_text(encoding="utf-8")
            self.assertIn("progress on stdout", log_text)
            self.assertIn("warning on stderr", log_text)
            self.assertIn("Traceback (most recent call last)", log_text)
            self.assertIn("RuntimeError: boom", log_text)

            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"], "RuntimeError: boom")
            records = _metrics_records(directory)
            self.assertEqual(records[-1]["event"], "session_end")
            self.assertEqual(records[-1]["status"], "failed")

    def test_keyboard_interrupt_marks_session_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(KeyboardInterrupt):
                with ExperimentLogger(
                    root, _config(tensorboard=False)
                ) as logger:
                    directory = logger.directory
                    raise KeyboardInterrupt
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "interrupted")
            self.assertIn("KeyboardInterrupt", metadata["error"])

    def test_sessions_are_isolated_even_with_the_same_output_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(tensorboard=False)
            with ExperimentLogger(root, config) as first:
                first.event("first_only")
                first.log_scalars(
                    "A", "train_epoch", {"loss": 1.0}, step=1, epoch=1
                )
                first_directory = first.directory
                first_session = first.session_id
            first_text = (first_directory / "metrics.jsonl").read_text(
                encoding="utf-8"
            )

            with ExperimentLogger(root, config) as second:
                second.event("second_only")
                second.log_scalars(
                    "B", "train_epoch", {"loss": 2.0}, step=1, epoch=1
                )
                second_directory = second.directory
                second_session = second.session_id
                self.assertNotEqual(first_directory, second_directory)

            self.assertNotEqual(first_session, second_session)
            self.assertTrue(first_directory.is_dir())
            self.assertTrue(second_directory.is_dir())
            # 第二次写入不能改动第一次会话的文件。
            self.assertEqual(
                (first_directory / "metrics.jsonl").read_text(encoding="utf-8"),
                first_text,
            )
            first_records = _metrics_records(first_directory)
            second_records = _metrics_records(second_directory)
            self.assertIn("first_only", [record["event"] for record in first_records])
            self.assertNotIn(
                "second_only", [record["event"] for record in first_records]
            )
            self.assertIn(
                "second_only", [record["event"] for record in second_records]
            )
            self.assertNotIn(
                "first_only", [record["event"] for record in second_records]
            )
            self.assertFalse(
                any(
                    record.get("stage") == "B"
                    for record in first_records
                    if record["event"] == "metrics"
                )
            )
            self.assertFalse(
                any(
                    record.get("stage") == "A"
                    for record in second_records
                    if record["event"] == "metrics"
                )
            )

    def test_logging_does_not_advance_python_numpy_or_torch_rng(self):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as temporary:
            with ExperimentLogger(Path(temporary), _config()) as logger:
                logger.event("rng", seed=1)
                logger.log_scalars(
                    "A", "train_batch", {"loss": 1.0}, step=1, epoch=1
                )
                logger.flush()

        self.assertEqual(random.getstate(), python_state)
        restored_numpy = np.random.get_state()
        self.assertEqual(restored_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
        self.assertEqual(restored_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_no_tensorboard_does_not_construct_a_writer_or_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "torch.utils.tensorboard.SummaryWriter",
                side_effect=AssertionError("SummaryWriter must not be constructed"),
            ):
                with ExperimentLogger(
                    root, _config(tensorboard=False)
                ) as logger:
                    self.assertIsNone(logger.writer)
                    logger.log_scalars(
                        "A", "train_batch", {"loss": 1.0}, step=1, epoch=1
                    )
                    directory = logger.directory
            self.assertFalse((directory / "tensorboard").exists())
            self.assertTrue(
                any(
                    record["event"] == "metrics"
                    for record in _metrics_records(directory)
                )
            )

    def test_summary_writer_receives_flush_secs_and_is_closed(self):
        created = {}

        class FakeWriter:
            def __init__(self, log_dir, flush_secs):
                created["log_dir"] = Path(log_dir)
                created["flush_secs"] = flush_secs
                self.closed = False
                self.scalars = []

            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def flush(self):
                created["flushed"] = True

            def close(self):
                self.closed = True
                created["closed"] = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("torch.utils.tensorboard.SummaryWriter", FakeWriter):
                with ExperimentLogger(
                    root, _config(flush_secs=7)
                ) as logger:
                    logger.log_scalars(
                        "C", "val_epoch", {"iou": 0.5}, step=1, epoch=1
                    )
                    writer = logger.writer
                    self.assertIsInstance(writer, FakeWriter)
            self.assertEqual(created["flush_secs"], 7)
            self.assertEqual(created["log_dir"], logger.directory / "tensorboard")
            self.assertEqual(writer.scalars, [("C/val_epoch/iou", 0.5, 1)])
            self.assertTrue(created.get("flushed"))
            self.assertTrue(created.get("closed"))

    def test_writer_initialization_failure_restores_streams_and_records_failure(self):
        original_stdout, original_stderr = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as temporary:
            logger = ExperimentLogger(Path(temporary), _config())
            with patch("torch.utils.tensorboard.SummaryWriter", side_effect=RuntimeError("writer failed")):
                with self.assertRaisesRegex(RuntimeError, "writer failed"):
                    with logger:
                        self.fail("writer initialization must fail before entering")
            metadata = json.loads((logger.directory / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("writer failed", (logger.directory / "train.log").read_text(encoding="utf-8"))
        self.assertIs(sys.stdout, original_stdout)
        self.assertIs(sys.stderr, original_stderr)

    def test_system_exit_status_distinguishes_interrupt_and_cli_failure(self):
        for code, expected in ((0, "completed"), (2, "failed"), (130, "interrupted")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                logger = ExperimentLogger(Path(temporary), _config(tensorboard=False))
                with self.assertRaises(SystemExit):
                    with logger:
                        raise SystemExit(code)
                metadata = json.loads((logger.directory / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], expected)

    def test_log_every_is_at_least_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = ExperimentLogger(root, _config(log_every=0, tensorboard=False))
            self.assertEqual(logger.log_every, 1)
            with logger:
                pass


if __name__ == "__main__":
    unittest.main()
