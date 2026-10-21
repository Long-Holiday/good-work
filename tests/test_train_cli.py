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


if __name__ == "__main__":
    unittest.main()
