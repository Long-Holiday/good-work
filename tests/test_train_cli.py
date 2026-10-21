"""Training CLI resume validation and experiment logging wiring."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import train
from diffpeak.engine import DiffPeakEngine
from diffpeak.inference import DEFAULT_CONFIG


def _mock_logger():
    logger = Mock()
    logger.__enter__ = Mock(return_value=logger)
    logger.__exit__ = Mock(return_value=False)
    return logger


class TrainingResumeBoundaryTest(unittest.TestCase):
    def test_completed_ddpo_stage_still_rejects_resume(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        payload = {"stage": "D", "epoch": 100, "format_version": 1}
        engine = Mock()
        engine.output_dir = Path("unused")
        logger = _mock_logger()
        bare = object.__new__(DiffPeakEngine)
        engine._validate_resume_algorithm.side_effect = bare._validate_resume_algorithm
        with patch.object(sys, "argv", ["train.py", "--resume", "old.pt"]), patch.object(
            train, "_load_training_config", return_value=(config, None, payload)
        ), patch.object(train, "DiffPeakEngine", return_value=engine), patch.object(
            train, "ExperimentLogger", return_value=logger
        ):
            with self.assertRaisesRegex(ValueError, "--init"):
                train.main()
        engine.load_models_from_payload.assert_not_called()
        engine.run_stage.assert_not_called()
        logger.__enter__.assert_called_once()
        logger.__exit__.assert_called_once()


class TrainingLoggingCliTest(unittest.TestCase):
    def _run_main(self, argv, config):
        engine = Mock()
        engine.output_dir = Path("unused")
        engine.device = "cpu"
        logger = _mock_logger()
        with patch.object(sys, "argv", ["train.py", *argv]), patch.object(
            train, "_load_training_config", return_value=(config, None, None)
        ), patch.object(
            train, "DiffPeakEngine", return_value=engine
        ) as engine_cls, patch.object(
            train, "ExperimentLogger", return_value=logger
        ) as logger_cls, patch.object(
            train, "_run_training"
        ) as run_training, patch.object(
            train, "seed_everything"
        ):
            train.main()
        return engine, engine_cls, logger, logger_cls, run_training

    def test_output_dir_and_experiment_name_build_a_named_absolute_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "outputs"
            config = copy.deepcopy(DEFAULT_CONFIG)
            engine, engine_cls, logger, logger_cls, run_training = self._run_main(
                ["--output-dir", str(out), "--experiment-name", "exp-1"], config
            )

            expected = out.resolve() / "exp-1"
            call_args, call_kwargs = logger_cls.call_args
            self.assertEqual(Path(call_args[0]), expected)
            self.assertEqual(call_args[1]["output_dir"], str(expected))
            self.assertEqual(engine_cls.call_args.args[0]["output_dir"], str(expected))
            self.assertEqual(call_kwargs["metadata"]["requested_stages"], "all")
            self.assertIsNone(call_kwargs["metadata"]["resume"])
            self.assertIsNone(call_kwargs["metadata"]["init"])
            logger.save_config.assert_called_once_with(engine._resolved_config())
            logger.event.assert_any_call(
                "engine_ready", device=str(engine.device), seed=42
            )
            run_training.assert_called_once_with(
                ANY, ANY, config, None, engine, logger
            )

    def test_no_tensorboard_disables_logging_in_config_passed_to_engine(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["logging"] = {"tensorboard": True, "log_every": 10, "flush_secs": 30}
        _, engine_cls, _, logger_cls, _ = self._run_main(["--no-tensorboard"], config)

        _, call_kwargs = logger_cls.call_args
        passed_config = logger_cls.call_args.args[1]
        self.assertFalse(passed_config["logging"]["tensorboard"])
        self.assertFalse(config["logging"]["tensorboard"])
        self.assertFalse(engine_cls.call_args.args[0]["logging"]["tensorboard"])
        self.assertEqual(call_kwargs["metadata"]["requested_stages"], "all")

    def test_default_config_keeps_tensorboard_enabled(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        _, _, _, logger_cls, _ = self._run_main([], config)
        passed_config = logger_cls.call_args.args[1]
        self.assertTrue(passed_config["logging"]["tensorboard"])

    def test_invalid_experiment_name_is_rejected_before_starting_a_session(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        for name in ("", "..", "nested/name", "nested\\name"):
            with self.subTest(name=name):
                engine = Mock()
                logger_cls = Mock()
                with patch.object(
                    sys, "argv", ["train.py", "--experiment-name", name]
                ), patch.object(
                    train, "_load_training_config",
                    return_value=(config, None, None),
                ), patch.object(
                    train, "DiffPeakEngine", return_value=engine
                ), patch.object(
                    train, "ExperimentLogger", logger_cls
                ), patch.object(
                    train, "seed_everything"
                ), patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as raised:
                        train.main()
                self.assertEqual(raised.exception.code, 2)
                logger_cls.assert_not_called()
                engine.run_stage.assert_not_called()

    def test_experiment_name_cannot_be_combined_with_resume(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        logger_cls = Mock()
        with patch.object(
            sys, "argv",
            ["train.py", "--resume", "last.pt", "--experiment-name", "exp"],
        ), patch.object(
            train, "_load_training_config", return_value=(config, None, {})
        ), patch.object(
            train, "DiffPeakEngine", return_value=Mock()
        ), patch.object(
            train, "ExperimentLogger", logger_cls
        ), patch.object(
            train, "seed_everything"
        ), patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                train.main()
        self.assertEqual(raised.exception.code, 2)
        logger_cls.assert_not_called()


class TrainingThreadCliTest(unittest.TestCase):
    def _run_main(self, argv, config, actual_cpu=14, actual_interop=4):
        engine = Mock()
        engine.output_dir = Path("unused")
        engine.device = "cpu"
        logger = _mock_logger()
        events = []

        def make_engine(*args, **kwargs):
            events.append(("engine", None))
            return engine

        with patch.object(sys, "argv", ["train.py", *argv]), patch.object(
            train, "_load_training_config", return_value=(config, None, None)
        ), patch.object(
            train, "DiffPeakEngine", side_effect=make_engine
        ) as engine_cls, patch.object(
            train, "ExperimentLogger", return_value=logger
        ) as logger_cls, patch.object(
            train, "_run_training"
        ), patch.object(
            train, "seed_everything"
        ), patch.object(
            train.torch, "set_num_threads",
            side_effect=lambda value: events.append(("cpu_threads", value)),
        ) as set_num_threads, patch.object(
            train.torch, "set_num_interop_threads",
            side_effect=lambda value: events.append(("interop_threads", value)),
        ) as set_interop_threads, patch.object(
            train.torch, "get_num_threads", return_value=actual_cpu
        ), patch.object(
            train.torch, "get_num_interop_threads", return_value=actual_interop
        ):
            train.main()
        return (
            engine, engine_cls, logger_cls, events,
            set_num_threads, set_interop_threads,
        )

    def test_positive_thread_flags_are_applied_before_engine_and_logged(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        (
            engine, engine_cls, logger_cls, events,
            set_num_threads, set_interop_threads,
        ) = self._run_main(
            ["--cpu-threads", "14", "--interop-threads", "4"], config
        )

        # 两个 setter 都在 engine（模型构造）之前被调用一次。
        self.assertEqual(
            events,
            [("cpu_threads", 14), ("interop_threads", 4), ("engine", None)],
        )
        set_num_threads.assert_called_once_with(14)
        set_interop_threads.assert_called_once_with(4)
        engine_cls.assert_called_once()

        metadata = logger_cls.call_args.kwargs["metadata"]
        self.assertEqual(metadata["cpu_threads_requested"], 14)
        self.assertEqual(metadata["cpu_threads_actual"], 14)
        self.assertEqual(metadata["interop_threads_requested"], 4)
        self.assertEqual(metadata["interop_threads_actual"], 4)

    def test_absent_thread_flags_keep_defaults_and_log_actual_counts(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        (
            _, _, logger_cls, events,
            set_num_threads, set_interop_threads,
        ) = self._run_main([], config, actual_cpu=56, actual_interop=2)

        # 未传 flag 时不触碰 PyTorch 设置，只记录实际生效的默认值。
        self.assertEqual(events, [("engine", None)])
        set_num_threads.assert_not_called()
        set_interop_threads.assert_not_called()

        metadata = logger_cls.call_args.kwargs["metadata"]
        self.assertIsNone(metadata["cpu_threads_requested"])
        self.assertEqual(metadata["cpu_threads_actual"], 56)
        self.assertIsNone(metadata["interop_threads_requested"])
        self.assertEqual(metadata["interop_threads_actual"], 2)

    def test_thread_flags_reject_non_positive_values(self):
        parser = train.build_parser()
        for flag in ("--cpu-threads", "--interop-threads"):
            for value in ("0", "-1", "abc"):
                with self.subTest(flag=flag, value=value):
                    with patch("sys.stderr"):
                        with self.assertRaises(SystemExit) as raised:
                            parser.parse_args([flag, value])
                    self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
