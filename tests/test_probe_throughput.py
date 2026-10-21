"""CPU checks for portable four-stage throughput orchestration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('throughput', ROOT / 'scripts/probe_throughput.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / 'data/train'
        data.mkdir(parents=True)
        for name in ('one', 'two', 'three'):
            (data / f'{name}_sat.jpg').touch()
            (data / f'{name}_mask.png').touch()
        (self.root / 'train.json').write_text(json.dumps(['train/one_sat.jpg', 'train/two_sat.jpg']))
        (self.root / 'val.json').write_text(json.dumps(['train/three_sat.jpg']))
        self.config = self.root / 'config.yaml'
        self.config.write_text(yaml.safe_dump(dict(data=dict(root=str(data.parent), train_split=str(self.root / 'train.json'), val_split=str(self.root / 'val.json')), model=dict(sam_checkpoint=str(self.root / 'missing.pt')), train=dict(batch_size=1, accumulation_steps=4), rl=dict(group_size=4, point_cost_warmup_steps=100), diffusion=dict(sampling_steps=20))))

    def args(self, *extra):
        return suite.parser().parse_args(['--config', str(self.config), '--output-dir', str(self.root / 'out'), '--train-samples', '2', '--batches', '1', '2', '--bootstrap', '--dry-run', *extra])

    def test_four_stages_and_bootstrap_share_parent(self):
        plan = suite.build_plan(self.args())
        self.assertEqual(plan['stages'], list('ABCD'))
        self.assertFalse((self.root / 'out').exists())
        for stage in 'BCD':
            rows = [r for r in plan['candidates'] if r['stage'] == stage]
            self.assertEqual(len({r['parent'] for r in rows}), 1)
            self.assertIn('/' + chr(ord(stage) - 1) + '/', rows[0]['parent'])
        for row in plan['candidates']:
            if row['stage'] != 'D':
                self.assertEqual(row['batch_size'] * row['accumulation_steps'], 4)
            else:
                self.assertNotIn('--accumulation-steps', row['command'])
                self.assertNotIn('--group-size', row['command'])
                self.assertNotIn('--sampling-steps', row['command'])

    def test_parent_required_and_strict_checkpoint_preflight(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            suite.build_plan(self.args('--stages', 'C', 'D'))
        with self.assertRaises(FileNotFoundError):
            suite.build_plan(self.args('--check-inputs'))

    def test_reject_overlap_escape_and_d_rounding(self):
        (self.root / 'val.json').write_text(json.dumps(['train/one_sat.jpg']))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            suite.build_plan(self.args())
        (self.root / 'val.json').write_text(json.dumps(['../escape_sat.jpg']))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            suite.build_plan(self.args())
        with self.assertRaisesRegex(ValueError, 'divide both'):
            suite.build_plan(self.args('--d-batches', '3'))

    def result_row(self, name='baseline', speed=10, skipped=0, repeat=0):
        directory = self.root / f'{name}{repeat}'
        directory.mkdir()
        (directory / 'results.json').write_text(json.dumps(dict(status='ok', parameters=dict(all_finite=True), measurement=dict(all_metrics_finite=True, optimizer=dict(successful_updates=4, skipped_updates=skipped), throughput=dict(samples_per_second=speed)))))
        (directory / 'splits.json').write_text(json.dumps({s: dict(subset_entries_sha256=s) for s in ('train', 'val')}))
        return suite.assess(dict(stage='A', name=name, repeat=repeat, directory=str(directory), batch_size=1, accumulation_steps=4, workers=4, checkpointing='off'), 0)

    def test_ranking_requires_healthy_complete_repeats(self):
        rows = [self.result_row(speed=8), self.result_row(speed=12, repeat=1), self.result_row('fast_but_skipped', 100, 1)]
        selected = suite.summarize(rows, 2)['A']['selected']
        self.assertEqual(selected['name'], 'baseline')
        self.assertEqual(selected['median_samples_per_second'], 10)
        rows[1]['split_key'] = ('different', 'val')
        self.assertIsNone(suite.summarize(rows, 2)['A']['selected'])

    def test_execution_preserves_failures_and_exports_config(self):
        args = self.args('--stages', 'A', '--workers', '0', '--checkpointing', 'off', '--batches', '1', '--repeats', '1')
        plan = suite.build_plan(args)
        def fake_run(command, *unused):
            directory = Path(command[command.index('--output-dir') + 1])
            directory.mkdir(parents=True)
            (directory / 'results.json').write_text(json.dumps(dict(status='ok', parameters=dict(all_finite=True), measurement=dict(all_metrics_finite=True, samples=64, optimizer=dict(successful_updates=4, skipped_updates=0), throughput=dict(samples_per_second=10)))))
            (directory / 'splits.json').write_text(json.dumps({s: dict(subset_entries_sha256=s) for s in ('train', 'val')}))
            return 0
        with mock.patch.object(suite, 'run_candidate', side_effect=fake_run):
            self.assertEqual(suite.execute(plan, args), 0)
        self.assertTrue(json.loads((self.root / 'out/completion.json').read_text())['complete'])
        config = yaml.safe_load((self.root / 'out/recommended_A.yaml').read_text())
        self.assertEqual(config['train']['batch_size'], 1)
        with self.assertRaises(FileExistsError):
            suite.build_plan(args)

    def test_failed_bootstrap_parent_blocks_downstream_execution(self):
        args = self.args('--stages', 'A', 'B', '--workers', '0', '--checkpointing', 'off', '--batches', '1', '--repeats', '1')
        plan = suite.build_plan(args)
        def fail(command, *unused):
            directory = Path(command[command.index('--output-dir') + 1])
            directory.mkdir(parents=True)
            (directory / 'weights.pt').touch()
            return 1
        with mock.patch.object(suite, 'run_candidate', side_effect=fail) as run:
            self.assertEqual(suite.execute(plan, args), 1)
        self.assertEqual(run.call_count, 1)
        rows = [json.loads(line) for line in (self.root / 'out/suite.jsonl').read_text().splitlines()]
        self.assertEqual(rows[1]['returncode'], 125)
        self.assertFalse((self.root / 'out/recommended_B.yaml').exists())

    def test_timeout_kills_candidate_process_group(self):
        process = mock.MagicMock()
        process.pid = 12345
        process.wait.side_effect = [suite.subprocess.TimeoutExpired('probe', 1), 0]
        process.__enter__.return_value = process
        with mock.patch.object(suite.subprocess, 'Popen', return_value=process) as popen, mock.patch.object(suite.os, 'killpg') as kill:
            with self.assertRaises(suite.subprocess.TimeoutExpired):
                suite.run_candidate(['probe'], None, {}, 1)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(12345, suite.signal.SIGKILL)
        self.assertEqual(process.wait.call_count, 2)

    def test_d_amp_skips_excluded(self):
        row = self.result_row()
        directory = Path(row['directory'])
        result = json.loads((directory / 'results.json').read_text())
        result['measurement']['optimizer'] = dict(successful_updates=4, amp_skipped_updates=1)
        (directory / 'results.json').write_text(json.dumps(result))
        self.assertIn('skipped_updates', suite.assess(row | {'stage': 'D'}, 0)['reasons'])


if __name__ == '__main__':
    unittest.main()
