"""Training CLI must validate old D snapshots even when the stage is complete."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import train
from diffpeak.engine import DiffPeakEngine
from diffpeak.inference import DEFAULT_CONFIG


class TrainingResumeBoundaryTest(unittest.TestCase):
    def test_completed_ddpo_stage_still_rejects_resume(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        payload = {"stage": "D", "epoch": 100, "format_version": 1}
        engine = Mock()
        engine.output_dir = Path("unused")
        bare = object.__new__(DiffPeakEngine)
        engine._validate_resume_algorithm.side_effect = bare._validate_resume_algorithm
        with patch.object(sys, "argv", ["train.py", "--resume", "old.pt"]), patch.object(
            train, "_load_training_config", return_value=(config, None, payload)
        ), patch.object(train, "DiffPeakEngine", return_value=engine):
            with self.assertRaisesRegex(ValueError, "--init"):
                train.main()
        engine.load_models_from_payload.assert_not_called()
        engine.run_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
