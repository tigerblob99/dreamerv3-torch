import importlib
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


def _build_import_stubs():
    stubs = {}

    ruamel_mod = types.ModuleType("ruamel")
    ruamel_mod.__path__ = []
    yaml_mod = types.ModuleType("ruamel.yaml")
    ruamel_mod.yaml = yaml_mod
    stubs["ruamel"] = ruamel_mod
    stubs["ruamel.yaml"] = yaml_mod

    stubs["wandb"] = types.ModuleType("wandb")
    stubs["gym"] = types.ModuleType("gym")

    tensorboard_mod = types.ModuleType("torch.utils.tensorboard")

    class _SummaryWriter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    tensorboard_mod.SummaryWriter = _SummaryWriter
    stubs["torch.utils.tensorboard"] = tensorboard_mod

    parallel_mod = types.ModuleType("parallel")

    class _Parallel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    parallel_mod.Parallel = _Parallel
    stubs["parallel"] = parallel_mod

    sweep_mod = types.ModuleType("BC_Sweep")
    sweep_mod._load_env_block = None
    stubs["BC_Sweep"] = sweep_mod

    bc_pkg = types.ModuleType("bc_mlp")
    bc_pkg.__path__ = []
    stubs["bc_mlp"] = bc_pkg

    eval_mod = types.ModuleType("bc_mlp.BC_MLP_eval")
    eval_mod._make_robomimic_env = None
    eval_mod._prepare_obs = None

    class _EpisodeVideoRecorder:
        pass

    eval_mod.EpisodeVideoRecorder = _EpisodeVideoRecorder
    eval_mod._extract_success = None
    stubs["bc_mlp.BC_MLP_eval"] = eval_mod
    return stubs


with mock.patch.dict(sys.modules, _build_import_stubs()):
    sys.modules.pop("joint_train", None)
    joint_train = importlib.import_module("joint_train")


class JointDatasetSamplingTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = pathlib.Path(self.tmpdir.name)
        for index in range(2):
            self._write_episode(f"episode_{index}")

    def _write_episode(self, name):
        episode = {
            "image": np.arange(3, dtype=np.uint8).reshape(3, 1, 1, 1),
            "action": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
            "reward": np.zeros((3,), dtype=np.float32),
            "discount": np.ones((3,), dtype=np.float32),
            "is_first": np.array([True, False, False]),
            "is_terminal": np.zeros((3,), dtype=np.float32),
            "robot0_joint_pos": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        }
        np.savez(self.data_dir / f"{name}.npz", **episode)

    def _config(self, **overrides):
        config = dict(
            batch_length=4,
            image_crop_height=0,
            image_crop_width=0,
            dataset_size=0,
            wm_random_crop=False,
            bc_random_crop=False,
            dreamer_like_sequence_sampling=False,
        )
        config.update(overrides)
        return SimpleNamespace(**config)

    def test_legacy_sampling_can_append_from_mid_episode(self):
        dataset = joint_train.JointDataset(
            self.data_dir,
            self._config(dreamer_like_sequence_sampling=False),
            mode="train",
        )
        with mock.patch.object(
            joint_train.np.random,
            "randint",
            side_effect=[0, 2, 0, 1, 0, 1],
        ):
            sample = dataset[0]

        self.assertEqual(sample["action"].squeeze(-1).tolist(), [2.0, 1.0, 2.0, 1.0])
        self.assertEqual(sample["is_first"].tolist(), [True, True, False, True])

    def test_dreamer_like_sampling_appends_from_episode_start(self):
        dataset = joint_train.JointDataset(
            self.data_dir,
            self._config(dreamer_like_sequence_sampling=True),
            mode="train",
        )
        with mock.patch.object(
            joint_train.np.random,
            "randint",
            side_effect=[0, 2, 0],
        ):
            sample = dataset[0]

        self.assertEqual(sample["action"].squeeze(-1).tolist(), [2.0, 0.0, 1.0, 2.0])

    def test_dreamer_like_is_first_marks_only_boundaries(self):
        dataset = joint_train.JointDataset(
            self.data_dir,
            self._config(dreamer_like_sequence_sampling=True),
            mode="train",
        )
        with mock.patch.object(
            joint_train.np.random,
            "randint",
            side_effect=[0, 2, 0],
        ):
            sample = dataset[0]

        self.assertEqual(sample["is_first"].tolist(), [True, True, False, False])

    def test_dreamer_like_zeroes_bc_targets_at_boundaries_and_final_step(self):
        dataset = joint_train.JointDataset(
            self.data_dir,
            self._config(dreamer_like_sequence_sampling=True),
            mode="train",
        )
        with mock.patch.object(
            joint_train.np.random,
            "randint",
            side_effect=[0, 2, 0],
        ):
            sample = dataset[0]

        self.assertEqual(sample["policy_target"].squeeze(-1).tolist(), [0.0, 1.0, 2.0, 0.0])
        self.assertEqual(sample["bc_mask"].tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_play_dataset_keeps_bc_mask_zero(self):
        dataset = joint_train.PlayDataDataset(
            self.data_dir,
            self._config(dreamer_like_sequence_sampling=True),
            mode="train",
        )
        with mock.patch.object(
            joint_train.np.random,
            "randint",
            side_effect=[0, 2, 0],
        ):
            sample = dataset[0]

        self.assertTrue(torch.equal(sample["bc_mask"], torch.zeros_like(sample["bc_mask"])))


if __name__ == "__main__":
    unittest.main()
