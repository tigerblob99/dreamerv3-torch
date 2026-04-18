import argparse
import os
import pathlib
import sys
import time
import re
import numpy as np
import ruamel.yaml as yaml
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gym
from collections import defaultdict
from types import SimpleNamespace


sys.path.append(str(pathlib.Path(__file__).parent.parent))

import tools
import networks
from models import WorldModel
import wandb

# Import Eval helpers
from parallel import Parallel
from bc_mlp.BC_MLP_eval import (
    _make_robomimic_env, 
    _prepare_obs, 
    EpisodeVideoRecorder, 
    _extract_success
)

# Import helper from BC_Sweep

from BC_Sweep import _load_env_block

""""python joint_train.py --configs joint_train robomimic --offline_traindir datasets/robomimic_data_MV/can_PH_train --offline_evaldir datasets/robomimic_data_MV/can_PH_eval --offline_playdir datasets/robomimic_data_MV/can_MH_train --logdir logdir/joint_Play_stoc_actr_2 --env_config can_env_eval"""

"""docker exec -it -w /workspace/dreamerv3-torch/dreamerv3-torch pytorch_dev_cu130 \
  bash -lc 'python joint_train.py --configs joint_train robomimic --offline_traindir datasets/robomimic_data_MV/can_PH_train --offline_evaldir datasets/robomimic_data_MV/can_PH_eval --offline_playdir datasets/robomimic_data_MV/can_MH_train --logdir logdir/joint_Play_1 --env_config can_env_eval 2>&1 | tee /workspace/eval.log'
"""


_EXCLUDE_KEYS = {"action", "reward", "discount", "is_first", "is_terminal", "policy_target"}

def to_np(tensor):
    return tensor.detach().cpu().numpy()


def _apply_action_mlp_overrides(config):
    action_cfg = getattr(config, "action_mlp", None)
    if action_cfg is None:
        return
    if hasattr(config, "action_mlp_layers"):
        action_cfg = dict(action_cfg)
        action_cfg["layers"] = int(config.action_mlp_layers)
        config.action_mlp = action_cfg

# --- Custom Action MLP (Distribution Wrapper) ---
class ActionMLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers=4,
        units=1024,
        act="SiLU",
        norm=False,
        dist="normal",
        std="learned",
        min_std=0.1,
        max_std=1.0,
        absmax=1.0,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cpu",
    ):
        super().__init__()
        self._mlp = networks.MLP(
            inp_dim,
            shape,
            layers,
            units,
            act,
            norm,
            dist,
            std,
            min_std,
            max_std,
            absmax=absmax,
            temp=temp,
            unimix_ratio=unimix_ratio,
            outscale=outscale,
            symlog_inputs=symlog_inputs,
            device=device,
            name="ActionMLP",
        )
        if device is not None:
            self.to(device)

    def forward(self, features, return_dist=False, sample=False):
        dist = self._mlp(features)
        if return_dist:
            return dist
        if sample:
            return dist.sample()
        return dist.mode()

# --- Dataset ---

class JointDataset(Dataset):
    def __init__(self, directory, config, mode='train', bc_mask_value=1.0):
        self.config = config
        self.mode = mode
        if directory is None:
            self.directories = []
        elif isinstance(directory, str):
            self.directories = [
                pathlib.Path(part.strip()).expanduser()
                for part in directory.split(",")
                if part.strip()
            ]
        elif isinstance(directory, (list, tuple)):
            self.directories = [
                pathlib.Path(part).expanduser()
                for part in directory
                if str(part).strip()
            ]
        else:
            self.directories = [pathlib.Path(directory).expanduser()]
        self.directory = self.directories[0] if len(self.directories) == 1 else self.directories
        self.batch_length = config.batch_length
        self.bc_mask_value = float(bc_mask_value)
        self.crop_h = config.image_crop_height
        self.crop_w = config.image_crop_width
        self.do_crop = (self.crop_h > 0 and self.crop_w > 0)
        self.orig_h = 84
        self.orig_w = 84

        self.episodes = {}
        total_steps = 0
        total_transitions = 0
        for dataset_dir in self.directories:
            if not dataset_dir.exists():
                print(f"Warning: Dataset directory {dataset_dir} does not exist.")
                continue
            remaining = None if not config.dataset_size else max(int(config.dataset_size) - total_transitions, 0)
            if remaining == 0:
                break
            episodes = tools.load_episodes(dataset_dir, limit=remaining)
            for episode_name, episode in episodes.items():
                key = episode_name if len(self.directories) == 1 else f"{dataset_dir}::{episode_name}"
                self.episodes[key] = episode
                total_steps += len(episode["action"])
                total_transitions += len(episode["reward"]) - 1
                if config.dataset_size and total_transitions >= config.dataset_size:
                    break
            if config.dataset_size and total_transitions >= config.dataset_size:
                break

        self.episode_list = list(self.episodes.values())
        self.sampleable_episodes = [
            episode for episode in self.episode_list if len(episode["action"]) > 1
        ]

        self.num_episodes = len(self.episode_list)
        self.num_sampleable_episodes = len(self.sampleable_episodes)
        self.epoch_size = (total_steps // self.batch_length) if self.batch_length else 0
        if self.sampleable_episodes:
            sample_episode = self.sampleable_episodes[0]
            image_tail = sample_episode["image"].shape[1:]
            if self.do_crop:
                image_tail = (self.crop_h, self.crop_w, image_tail[-1])
            self.image_shape = (self.batch_length, *image_tail)
            self.image_dtype = sample_episode["image"].dtype
            self.action_shape = (self.batch_length, *sample_episode["action"].shape[1:])
            self.action_dtype = sample_episode["action"].dtype
            self.is_first_dtype = sample_episode["is_first"].dtype
            self.is_terminal_dtype = sample_episode["is_terminal"].dtype
            self.extra_keys = [
                key
                for key in sample_episode.keys()
                if key not in {"image", "action", "is_first", "is_terminal"}
                and not key.startswith("log_")
            ]
            self.extra_specs = {
                key: (
                    (self.batch_length, *sample_episode[key].shape[1:]),
                    sample_episode[key].dtype,
                )
                for key in self.extra_keys
            }
        else:
            self.image_shape = None
            self.image_dtype = None
            self.action_shape = None
            self.action_dtype = None
            self.is_first_dtype = None
            self.is_terminal_dtype = None
            self.extra_keys = []
            self.extra_specs = {}
        
        print(f"[{mode}] Loaded {self.num_episodes} episodes, {total_steps} steps.")

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        idx = np.random.randint(0, self.num_sampleable_episodes)
        return self.sampleable_episodes[idx]

    def _get_crop_coords(self, use_random_crop):
        if not self.do_crop: return 0, 0
        if self.mode == 'train' and use_random_crop:
            top = np.random.randint(0, self.orig_h - self.crop_h + 1)
            left = np.random.randint(0, self.orig_w - self.crop_w + 1)
        else:
            top = (self.orig_h - self.crop_h) // 2
            left = (self.orig_w - self.crop_w) // 2
        return top, left

    def _crop(self, img, top, left):
        if not self.do_crop: return img
        return img[:, top : top + self.crop_h, left : left + self.crop_w, :]

    def __getitem__(self, _):
        dreamer_like = bool(
            getattr(self.config, "dreamer_like_sequence_sampling", False)
        )
        image_wm = np.empty(self.image_shape, dtype=self.image_dtype)
        share_crop = (not self.do_crop) or (
            self.mode != "train"
            or (
                not getattr(self.config, "wm_random_crop", False)
                and not getattr(self.config, "bc_random_crop", False)
            )
        )
        image_bc = image_wm if share_crop else np.empty(self.image_shape, dtype=self.image_dtype)
        action = np.empty(self.action_shape, dtype=self.action_dtype)
        policy_target = np.zeros(self.action_shape, dtype=self.action_dtype)
        is_first = np.empty((self.batch_length,), dtype=self.is_first_dtype)
        is_terminal = np.empty((self.batch_length,), dtype=self.is_terminal_dtype)
        bc_mask = np.zeros((self.batch_length,), dtype=np.float32)
        extra = {
            key: np.empty(shape, dtype=dtype)
            for key, (shape, dtype) in self.extra_specs.items()
        }
        current_len = 0
        
        while current_len < self.batch_length:
            episode = self._sample_episode()
            total_ep_len = len(episode['action'])

            needed = self.batch_length - current_len
            if dreamer_like and current_len > 0:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, total_ep_len)
            take = min(needed, total_ep_len - start_idx)
            if take <= 0: continue

            input_slice = slice(start_idx, start_idx + take)
            out_slice = slice(current_len, current_len + take)

            raw_imgs = episode['image'][input_slice]
            
            t_wm, l_wm = self._get_crop_coords(getattr(self.config, 'wm_random_crop', False))
            image_wm[out_slice] = self._crop(raw_imgs, t_wm, l_wm)
            if not share_crop:
                t_bc, l_bc = self._get_crop_coords(getattr(self.config, 'bc_random_crop', False))
                image_bc[out_slice] = self._crop(raw_imgs, t_bc, l_bc)

            action_chunk = episode['action'][input_slice]
            next_actions = episode['action'][start_idx + 1 : min(start_idx + take + 1, total_ep_len)]
            action[out_slice] = action_chunk
            if len(next_actions):
                policy_target[current_len : current_len + len(next_actions)] = next_actions
            bc_mask[out_slice] = self.bc_mask_value
            bc_mask[current_len + len(next_actions) : current_len + take] = 0.0

            is_first[out_slice] = episode['is_first'][input_slice]
            is_first[current_len] = True
            is_terminal[out_slice] = episode['is_terminal'][input_slice]

            for key in self.extra_keys:
                extra[key][out_slice] = episode[key][input_slice]

            current_len += take

        out = {
            'image_wm': torch.from_numpy(image_wm),
            'image_bc': torch.from_numpy(image_bc),
            'action': torch.from_numpy(action),
            'policy_target': torch.from_numpy(policy_target),
            'is_first': torch.from_numpy(is_first),
            'is_terminal': torch.from_numpy(is_terminal),
            'bc_mask': torch.from_numpy(bc_mask),
        }
        for key, value in extra.items():
            out[key] = torch.from_numpy(value)
        if dreamer_like and out["policy_target"].shape[0] > 0:
            boundary_mask = torch.zeros(out["bc_mask"].shape[0], dtype=torch.bool)
            if boundary_mask.numel() > 1:
                boundary_mask[:-1] = out["is_first"][1:] > 0
            boundary_mask[-1] = True
            out["policy_target"][boundary_mask] = 0
            out["bc_mask"][boundary_mask] = 0
        return out

class PlayDataDataset(JointDataset):
    def __init__(self, directory, config, mode='train'):
        super().__init__(directory, config, mode=mode, bc_mask_value=0.0)

def collate_episodes(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

# --- Evaluation Functions ---

def evaluate_offline(wm, policy, eval_loader, config, step, bc_eval=True, prefix="eval"):
    wm.eval()
    if bc_eval:
        policy.eval()
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for i, raw_batch in enumerate(eval_loader):
            if i >= config.offline_eval_batches: break
            
            data_wm = raw_batch.copy()
            data_wm['image'] = data_wm.pop('image_wm')
            data_wm = wm.preprocess(data_wm)
            
            embed = wm.encoder(data_wm)
            post, prior = wm.dynamics.observe(embed, data_wm['action'], data_wm['is_first'])
            feat = wm.dynamics.get_feat(post)
            
            kl_loss, kl_value, _, _ = wm.dynamics.kl_loss(
                post, prior, config.kl_free, config.dyn_scale, config.rep_scale
            )
            metrics[f"{prefix}/kl"].append(kl_value.mean().item())
            
            for name, head in wm.heads.items():
                pred = head(feat)
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        loss = -v.log_prob(data_wm[k])
                        metrics[f"{prefix}/{k}_loss"].append(loss.mean().item())
                else:
                    loss = -pred.log_prob(data_wm[name])
                    metrics[f"{prefix}/{name}_loss"].append(loss.mean().item())

            if bc_eval:
                img_bc = raw_batch['image_bc'].to(config.device, dtype=torch.float32, non_blocking=True) / 255.0
                if config.image_standardize and 'image_mean' in data_wm:
                     img_bc = (img_bc - data_wm['image_mean']) / data_wm['image_std']
                
                data_bc = data_wm.copy()
                data_bc['image'] = img_bc
                embed_bc = wm.encoder(data_bc)
                post_bc, _ = wm.dynamics.observe(embed_bc, data_bc['action'], data_bc['is_first'])
                feat_bc = wm.dynamics.get_feat(post_bc)
                
                # --- BC Loss (exact faithful loss when available, otherwise NLL) ---
                target = raw_batch['policy_target'].to(config.device, dtype=torch.float32, non_blocking=True)
                pred_dist = policy(feat_bc, return_dist=True)
                bc_loss = tools.regression_loss(pred_dist, target).mean()
                metrics[f"{prefix}/bc_loss"].append(bc_loss.item())

    agg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return agg_metrics

# --- GLOBAL WORKER CLASS (Moved outside evaluate_online to support pickling/reuse) ---
class EnvWorker:
    def __init__(self, env_cfg, img_size):
        self._cfg = env_cfg
        self._image_size = img_size
        self._env = _make_robomimic_env(self._cfg, self._image_size)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        return obs, reward, done, info, _extract_success(info, self._env)
    
    def set_state(self,state):
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        self._env.reset()
        self._env.sim.set_state_from_flattened(state)
        self._env.sim.forward()
        raw_obs = self._env._get_observations(force_update=True)
        return raw_obs

    def close(self):
        if self._env: self._env.close()

def evaluate_online(wm, policy, config, step, run, envs):
    """
    Evaluates policy online using persistent environments 'envs'.
    """
    wm.eval()
    policy.eval()
    
    num_envs = len(envs)
    total_episodes = config.eval_episodes
    max_video_episodes = max(0, int(getattr(config, "eval_video_episodes", 3)))
    
    # Handle optional video recording (skip if logdir is None)
    video_dir = None
    if config.logdir is not None and max_video_episodes > 0:
        video_dir = pathlib.Path(config.logdir) / "eval_videos" / f"step_{step}"
    
    # We use the envs passed in, we do NOT create/close them here.
    
    obs_batch = [None] * num_envs
    rssm_state = None 
    prev_action = torch.zeros((num_envs, config.num_actions), device=config.device)
    is_first = torch.ones(num_envs, device=config.device)
    
    episode_rewards = [0.0] * num_envs
    episode_steps = [0] * num_envs
    episode_success = [False] * num_envs
    completed_episodes = 0
    
    final_rewards = []
    final_successes = []
    final_reward_per_step = []
    
    recorders = [
        EpisodeVideoRecorder(
            directory=video_dir,
            fps=20,
            camera_key=config.camera_obs_keys[0],
            camera_keys=config.camera_obs_keys,
            flip_keys=config.flip_camera_keys
        ) for _ in range(num_envs)
    ] if video_dir else [None] * num_envs

    global_ep_idx = 0
    env_episode_ids = [0] * num_envs

    def _should_record(episode_idx):
        return video_dir is not None and episode_idx < max_video_episodes
    
    # Initial Reset
    print(f"Starting Online Eval ({total_episodes} episodes)...")
    for i in range(num_envs):
        if global_ep_idx < total_episodes:
            obs_batch[i] = envs[i].reset()() 
            if _should_record(global_ep_idx):
                recorders[i].start_episode(global_ep_idx)
                recorders[i].add_frame(obs_batch[i])
            env_episode_ids[i] = global_ep_idx
            global_ep_idx += 1
        else:
            obs_batch[i] = None

    while completed_episodes < total_episodes:
        active_indices = [i for i, obs in enumerate(obs_batch) if obs is not None]
        if not active_indices: break

        full_batch_lists = defaultdict(list)
        template_idx = active_indices[0]
        template_processed = _prepare_obs(
            obs_batch[template_idx],
            cnn_keys_order=config.bc_cnn_keys_order,
            mlp_keys_order=config.bc_mlp_keys_order,
            camera_keys=config.camera_obs_keys,
            flip_keys=config.flip_camera_keys,
            crop_height=config.image_crop_height if config.image_crop_height > 0 else None,
            crop_width=config.image_crop_width if config.image_crop_width > 0 else None,
            config=config
        )

        for i in range(num_envs):
            if obs_batch[i] is None:
                for k, v in template_processed.items():
                    full_batch_lists[k].append(np.zeros_like(v))
            else:
                processed = _prepare_obs(
                    obs_batch[i],
                    cnn_keys_order=config.bc_cnn_keys_order,
                    mlp_keys_order=config.bc_mlp_keys_order,
                    camera_keys=config.camera_obs_keys,
                    flip_keys=config.flip_camera_keys,
                    crop_height=config.image_crop_height if config.image_crop_height > 0 else None,
                    crop_width=config.image_crop_width if config.image_crop_width > 0 else None,
                    config=config
                )
                for k, v in processed.items():
                    full_batch_lists[k].append(v)

        data = {}
        for k, v_list in full_batch_lists.items():
            arr = np.stack(v_list)
            tensor = torch.as_tensor(arr, device=config.device).float()
            if k == 'image':
                tensor = tensor / 255.0
                if config.image_standardize and wm._dataset_image_mean is not None:
                    tensor = (tensor - wm._dataset_image_mean) / wm._dataset_image_std
            data[k] = tensor

        with torch.no_grad():
            embed = wm.encoder(data) 
            post, _ = wm.dynamics.obs_step(rssm_state, prev_action, embed, is_first, sample=False)
            rssm_state = post
            feat = wm.dynamics.get_feat(post)
            
            # --- Policy Prediction (Raw) ---
            action_tensor = policy(feat)
            if config.clip_actions:
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            
            prev_action = action_tensor

        action_np = action_tensor.cpu().numpy()
        promises = []
        for i in range(num_envs):
            if obs_batch[i] is not None:
                promises.append(envs[i].step(action_np[i]))
            else:
                promises.append(None)

        for i, promise in enumerate(promises):
            if promise is None: continue
            
            obs, reward, done, info, success = promise()
            
            episode_rewards[i] += reward
            episode_steps[i] += 1
            episode_success[i] = (episode_success[i] or success)
            if _should_record(env_episode_ids[i]):
                recorders[i].add_frame(obs)
            
            env_done = done or (episode_steps[i] >= getattr(config, "max_env_steps", 500)) or success
            
            if env_done:
                completed_episodes += 1
                final_rewards.append(episode_rewards[i])
                final_reward_per_step.append(episode_rewards[i] / episode_steps[i] if episode_steps[i] > 0 else 0.0)
                final_successes.append(episode_success[i])
                
                vid_path = (
                    recorders[i].finish_episode()
                    if _should_record(env_episode_ids[i])
                    else None
                )
                if vid_path and env_episode_ids[i] < max_video_episodes:
                    if run: 
                        run.log({f"eval_online/video_{env_episode_ids[i]}": wandb.Video(str(vid_path), fps=20, format="mp4")}, commit=False)
                
                if global_ep_idx < total_episodes:
                    obs_batch[i] = envs[i].reset()()
                    if _should_record(global_ep_idx):
                        recorders[i].start_episode(global_ep_idx)
                        recorders[i].add_frame(obs_batch[i])
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    episode_success[i] = False
                    env_episode_ids[i] = global_ep_idx
                    global_ep_idx += 1
                    is_first[i] = 1.0
                    prev_action[i] = 0.0
                else:
                    obs_batch[i] = None
                    is_first[i] = 0.0 
            else:
                obs_batch[i] = obs
                is_first[i] = 0.0 

    metrics = {
        "eval_online/success_rate": np.mean(final_successes) if final_successes else 0.0,
        "eval_online/mean_return": np.mean(final_rewards) if final_rewards else 0.0,
        "eval_online/mean_return_per_step": np.mean(final_reward_per_step) if final_reward_per_step else 0.0
    }
    return metrics

# --- Main ---

def joint_train(config):
    tools.set_seed_everywhere(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "Dreamer_Joint"),
        entity=os.getenv("WANDB_ENTITY"),
        config=vars(config),
        dir=str(logdir),
        name=config.run_name if hasattr(config, 'run_name') else None
    )
    print(f"Logging to {logdir}")

    # --- Setup Datasets & Dataloaders ---
    total_batch_size = config.batch_size
    expert_batch_size = int(round(total_batch_size * config.expert_data_fraction))
    play_batch_size = total_batch_size - expert_batch_size

    train_dataset = JointDataset(config.offline_traindir, config, mode='train')
    train_loader = DataLoader(
        train_dataset, batch_size=expert_batch_size, shuffle=True, 
        num_workers=config.num_workers, collate_fn=collate_episodes, 
        pin_memory=True, drop_last=True
    )

    eval_config = SimpleNamespace(**vars(config))
    eval_config.dataset_size = int(getattr(config, "eval_dataset_size", config.dataset_size) or 0)
    eval_dataset = JointDataset(config.offline_evaldir, eval_config, mode='eval')
    eval_loader = DataLoader(
        eval_dataset, batch_size=total_batch_size, shuffle=False, 
        num_workers=0, collate_fn=collate_episodes, pin_memory=True
    )
    
    def cycle(loader):
        while True:
            for b in loader: yield b
    train_iter = cycle(train_loader)

    play_iter = None
    play_dir = getattr(config, 'offline_playdir', None)
    if play_dir:
        play_dataset = PlayDataDataset(play_dir, config, mode='train')
        if play_dataset.num_episodes > 0 and len(play_dataset) > 0:
            play_loader = DataLoader(
                play_dataset, batch_size=play_batch_size, shuffle=True,
                num_workers=config.num_workers, collate_fn=collate_episodes,
                pin_memory=True, drop_last=True
            )
            play_iter = cycle(play_loader)
        else:
            print("PlayData dataset is empty; skipping PlayData in training.")

    def concat_batches(a, b):
        if a.keys() != b.keys():
            missing_a = b.keys() - a.keys()
            missing_b = a.keys() - b.keys()
            raise KeyError(
                f"Batch keys mismatch. Missing in a: {sorted(missing_a)} "
                f"Missing in b: {sorted(missing_b)}"
            )
        return {k: torch.cat([a[k], b[k]], dim=0) for k in a.keys()}

    print("Inferring Observation Space...")
    sample_ep = train_dataset.episode_list[0]
    obs_space, act_space = tools._define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]
    
    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    
    # --- Initialize ActionMLP (Raw Policy) ---
    if config.dyn_discrete:
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
        feat_size = config.dyn_stoch + config.dyn_deter

    ac = config.action_mlp
    policy = ActionMLP(
        inp_dim=feat_size,
        shape=(config.num_actions,),
        layers=ac['layers'],
        units=ac['units'],
        act=ac['act'],
        norm=ac['norm'],
        dist=ac['dist'],
        std=ac['std'],
        min_std=ac['min_std'],
        max_std=ac['max_std'],
        absmax=ac['absmax'],
        temp=ac['temp'],
        unimix_ratio=ac['unimix_ratio'],
        outscale=ac['outscale'],
        symlog_inputs=ac['symlog_inputs'],
        device=config.device,
    )
    
    # Print Param Count for Sanity Check
    print(f"--- Parameter Check ---")
    print(f"Policy Params:      {sum(p.numel() for p in policy.parameters()):,}")
    print(f"World Model Params: {sum(p.numel() for p in wm.parameters()):,}")
    print(f"-----------------------")

    # When wm and bc cropping produce identical images (cropping disabled, or
    # both branches use center crop), the BC-branch encoder + RSSM pass is a
    # pure duplicate of the WM pass. Detect that case once and reuse feat_wm
    # for the policy to halve the encoder/RSSM cost per step.
    do_crop = (config.image_crop_height > 0 and config.image_crop_width > 0)
    crops_match = (not do_crop) or (
        not getattr(config, 'wm_random_crop', False)
        and not getattr(config, 'bc_random_crop', False)
    )
    if crops_match:
        print("Crops match (image_wm == image_bc) — sharing WM features with BC branch.")
    else:
        print("Crops differ — running separate WM and BC encoder passes.")

    wm_optimizer = torch.optim.AdamW(
        list(wm.parameters()),
        lr=config.model_lr,
        eps=config.opt_eps,
        weight_decay=config.weight_decay,
    )
    policy_optimizer = torch.optim.AdamW(
        list(policy.parameters()),
        lr=config.model_lr,
        eps=config.opt_eps,
        weight_decay=config.weight_decay,
    )

    def next_batch():
        raw_batch = next(train_iter)
        if play_iter is not None:
            raw_play = next(play_iter)
            raw_batch = concat_batches(raw_batch, raw_play)
        return raw_batch

    def save_ckpt(path, phase, global_step, joint_step):
        torch.save(
            {
                'wm': wm.state_dict(),
                'policy': policy.state_dict(),
                'wm_optimizer': wm_optimizer.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'phase': phase,
                'global_step': global_step,
                'joint_step': joint_step,
            },
            path,
        )

    def init_eval_envs():
        print(f"Initializing {config.num_envs} persistent Eval Envs...")
        crop_h = config.image_crop_height
        crop_w = config.image_crop_width
        image_hw = (crop_h, crop_w) if (crop_h > 0 and crop_w > 0) else (84, 84)

        env_config = SimpleNamespace(
            robosuite_task=getattr(config, "robosuite_task", "PickPlaceCan"),
            robosuite_robots=getattr(config, "robosuite_robots", ["Panda"]),
            robosuite_controller=getattr(config, "robosuite_controller", "OSC_POSE"),
            robosuite_reward_shaping=getattr(config, "robosuite_reward_shaping", False),
            robosuite_control_freq=getattr(config, "robosuite_control_freq", 20),
            max_env_steps=getattr(config, "max_env_steps", 500),
            ignore_done=getattr(config, "ignore_done", False),
            has_renderer=getattr(config, "has_renderer", False),
            has_offscreen_renderer=getattr(config, "has_offscreen_renderer", True),
            use_camera_obs=getattr(config, "use_camera_obs", True),
            camera_depths=getattr(config, "camera_depths", False),
            camera_obs_keys=tuple(config.camera_obs_keys),
            seed=config.seed,
            render=getattr(config, "render", False)
        )
        if hasattr(config, "controller_configs"):
            env_config.controller_configs = config.controller_configs

        envs = []
        for _ in range(config.num_envs):
            envs.append(
                Parallel(lambda cfg=env_config, hw=image_hw: EnvWorker(cfg, hw), "process")
            )
        return envs

    eval_envs = None
    try:
        global_step = 0
        joint_step = 0

        wm_pretrain_steps = int(getattr(config, "wm_pretrain_steps", 0))
        if wm_pretrain_steps > 0:
            print("Starting World Model Pretraining...")
            for _ in range(wm_pretrain_steps):
                wm.train()

                raw_batch = next_batch()
                data_wm = raw_batch.copy()
                data_wm['image'] = data_wm.pop('image_wm')
                data_wm = wm.preprocess(data_wm)

                embed_wm = wm.encoder(data_wm)
                post_wm, prior_wm = wm.dynamics.observe(embed_wm, data_wm['action'], data_wm['is_first'])

                kl_loss, kl_val, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post_wm, prior_wm, config.kl_free, config.dyn_scale, config.rep_scale
                )

                feat_wm = wm.dynamics.get_feat(post_wm)
                head_scales = {
                    "reward": config.reward_head["loss_scale"],
                    "cont": config.cont_head["loss_scale"],
                }
                recon_losses = 0
                head_loss_means = {}
                head_loss_scaled_means = {}
                for name, head in wm.heads.items():
                    scale = head_scales.get(name, 1.0)
                    if scale == 0.0:
                        continue
                    pred = head(feat_wm)
                    if isinstance(pred, dict):
                        for k, v in pred.items():
                            loss = -v.log_prob(data_wm[k])
                            recon_losses += scale * loss
                            head_loss_means[k] = loss.mean()
                            head_loss_scaled_means[k] = (scale * loss).mean()
                    else:
                        loss = -pred.log_prob(data_wm[name])
                        recon_losses += scale * loss
                        head_loss_means[name] = loss.mean()
                        head_loss_scaled_means[name] = (scale * loss).mean()

                wm_loss = (recon_losses + kl_loss).mean()

                wm_optimizer.zero_grad()
                wm_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(list(wm.parameters()), config.grad_clip)
                wm_optimizer.step()

                global_step += 1

                if global_step % config.log_every == 0:
                    log_data = {
                        "wm_pretrain/train/total_loss": wm_loss.item(),
                        "wm_pretrain/train/wm_loss": wm_loss.item(),
                        "wm_pretrain/train/wm_recon_loss": recon_losses.mean().item(),
                        "wm_pretrain/train/kl": kl_val.mean().item(),
                        "wm_pretrain/train/dyn_loss": dyn_loss.mean().item(),
                        "wm_pretrain/train/rep_loss": rep_loss.mean().item(),
                        "wm_pretrain/train/grad_norm": float(grad_norm),
                    }
                    for name, value in head_loss_means.items():
                        log_data[f"wm_pretrain/train/{name}_loss"] = value.item()
                    for name, value in head_loss_scaled_means.items():
                        log_data[f"wm_pretrain/train/{name}_loss_scaled"] = value.item()
                    wandb.log(log_data, step=global_step)

                if global_step % config.eval_every == 0:
                    print(f"Evaluating at step {global_step}...")
                    off_metrics = evaluate_offline(
                        wm, policy, eval_loader, config, global_step, bc_eval=False, prefix="wm_pretrain/eval"
                    )
                    wandb.log(off_metrics, step=global_step)
                    save_ckpt(logdir / "latest.pt", "wm_pretrain", global_step, joint_step)

                if global_step % config.save_every == 0:
                    save_ckpt(logdir / f"step_{global_step}.pt", "wm_pretrain", global_step, joint_step)

            save_ckpt(logdir / "wm_pretrain_end.pt", "wm_pretrain", global_step, joint_step)
            save_ckpt(logdir / "latest.pt", "wm_pretrain", global_step, joint_step)

        print("Starting Joint Pretraining...")
        for _ in range(int(config.steps)):
            wm.train()
            policy.train()

            raw_batch = next_batch()

            data_wm = raw_batch.copy()
            data_wm['image'] = data_wm.pop('image_wm')
            data_wm = wm.preprocess(data_wm)

            embed_wm = wm.encoder(data_wm)
            post_wm, prior_wm = wm.dynamics.observe(embed_wm, data_wm['action'], data_wm['is_first'])

            kl_loss, kl_val, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                post_wm, prior_wm, config.kl_free, config.dyn_scale, config.rep_scale
            )

            feat_wm = wm.dynamics.get_feat(post_wm)
            head_scales = {
                "reward": config.reward_head["loss_scale"],
                "cont": config.cont_head["loss_scale"],
            }
            recon_losses = 0
            head_loss_means = {}
            head_loss_scaled_means = {}
            for name, head in wm.heads.items():
                scale = head_scales.get(name, 1.0)
                if scale == 0.0:
                    continue
                pred = head(feat_wm)
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        loss = -v.log_prob(data_wm[k])
                        recon_losses += scale * loss
                        head_loss_means[k] = loss.mean()
                        head_loss_scaled_means[k] = (scale * loss).mean()
                else:
                    loss = -pred.log_prob(data_wm[name])
                    recon_losses += scale * loss
                    head_loss_means[name] = loss.mean()
                    head_loss_scaled_means[name] = (scale * loss).mean()

            wm_loss = (recon_losses + kl_loss).mean()

            if crops_match:
                feat_bc = feat_wm
            else:
                img_bc = raw_batch['image_bc'].to(config.device, dtype=torch.float32, non_blocking=True) / 255.0
                if config.image_standardize and 'image_mean' in data_wm:
                     img_bc = (img_bc - data_wm['image_mean']) / data_wm['image_std']
                data_bc = data_wm.copy()
                data_bc['image'] = img_bc
                embed_bc = wm.encoder(data_bc)
                post_bc, _ = wm.dynamics.observe(embed_bc, data_bc['action'], data_bc['is_first'])
                feat_bc = wm.dynamics.get_feat(post_bc)

            target = raw_batch['policy_target'].to(config.device, dtype=torch.float32, non_blocking=True)
            if config.bc_sg_wm:
                feat_bc = feat_bc.detach()
            pred_dist = policy(feat_bc, return_dist=True)
            per_step_bc_loss = tools.regression_loss(pred_dist, target)
            bc_mask = raw_batch['bc_mask'].to(config.device, dtype=torch.float32, non_blocking=True)
            bc_mask_sum = bc_mask.sum()
            if bc_mask_sum > 0:
                bc_loss = (per_step_bc_loss * bc_mask).sum() / bc_mask_sum
            else:
                bc_loss = torch.tensor(0.0, device=config.device)

            with torch.no_grad():
                policy_entropy = pred_dist.entropy().mean()
                bc_valid_frac = (bc_mask > 0).float().mean()
                pred_action = pred_dist.mode()
                per_step_action_mse = torch.mean(torch.square(pred_action - target), dim=-1)
                per_step_action_mae = torch.mean(torch.abs(pred_action - target), dim=-1)
                if bc_mask_sum > 0:
                    action_mse = (per_step_action_mse * bc_mask).sum() / bc_mask_sum
                    action_mae = (per_step_action_mae * bc_mask).sum() / bc_mask_sum
                else:
                    action_mse = torch.tensor(0.0, device=config.device)
                    action_mae = torch.tensor(0.0, device=config.device)

            total_loss = (config.wm_loss_scale * wm_loss) + (config.bc_loss_scale * bc_loss)

            wm_optimizer.zero_grad()
            policy_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                list(wm.parameters()) + list(policy.parameters()), config.grad_clip
            )
            wm_optimizer.step()
            policy_optimizer.step()

            global_step += 1
            joint_step += 1

            if global_step % config.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/wm_loss": wm_loss.item(),
                    "train/wm_recon_loss": recon_losses.mean().item(),
                    "train/bc_loss": bc_loss.item(),
                    "train/kl": kl_val.mean().item(),
                    "train/dyn_loss": dyn_loss.mean().item(),
                    "train/rep_loss": rep_loss.mean().item(),
                    "train/policy_entropy": policy_entropy.item(),
                    "train/action_mse": action_mse.item(),
                    "train/action_mae": action_mae.item(),
                    "train/bc_valid_frac": bc_valid_frac.item(),
                    "train/bc_valid_steps": bc_mask_sum.item(),
                    "train/grad_norm": float(grad_norm),
                }
                for name, value in head_loss_means.items():
                    log_data[f"train/{name}_loss"] = value.item()
                for name, value in head_loss_scaled_means.items():
                    log_data[f"train/{name}_loss_scaled"] = value.item()
                wandb.log(log_data, step=global_step)

            if global_step % config.eval_every == 0:
                print(f"Evaluating at step {global_step}...")
                off_metrics = evaluate_offline(wm, policy, eval_loader, config, global_step)
                wandb.log(off_metrics, step=global_step)

                if eval_envs is None:
                    eval_envs = init_eval_envs()
                on_metrics = evaluate_online(wm, policy, config, global_step, run, eval_envs)
                wandb.log(on_metrics, step=global_step)

                save_ckpt(logdir / "latest.pt", "joint", global_step, joint_step)

            if global_step % config.save_every == 0:
                save_ckpt(logdir / f"step_{global_step}.pt", "joint", global_step, joint_step)

        final_phase = "joint" if joint_step > 0 else "wm_pretrain"
        save_ckpt(logdir / "latest.pt", final_phase, global_step, joint_step)
        print("Pretraining Finished.")
    
    finally:
        print("Closing Eval Envs...")
        if eval_envs is not None:
            for env in eval_envs:
                try: env.close()
                except: pass

if __name__ == "__main__":
    # 1. Pre-parse to get the env_config name
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--configs", nargs="+")
    pre_parser.add_argument("--env_config", type=str, default=None, help="Name of env config block in configs.yaml")
    args, remaining = pre_parser.parse_known_args()
    
    # 2. Load Base Configs
    cfg_path = pathlib.Path(sys.argv[0]).parent / "configs.yaml"
    configs = yaml.safe_load(cfg_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # 3. Load and Merge Env Config (Using Import with Fallback)
    complex_defaults = {} 
    
    if args.env_config:
        print(f"Loading env config block: {args.env_config}")
        env_defaults = {}
        
        # Use imported helper if available
        if _load_env_block:
            try:
                env_defaults = _load_env_block(args.env_config)
            except FileNotFoundError:
                print("Warning: BC_Sweep._load_env_block failed to find configs.yaml (path mismatch). Using local fallback.")
                env_defaults = configs.get(args.env_config, {})
            except Exception as e:
                print(f"Warning: Error in _load_env_block: {e}")
                env_defaults = {}
        else:
            env_defaults = configs.get(args.env_config, {})
        
        if env_defaults:
            for k, v in env_defaults.items():
                if isinstance(v, (dict, list)):
                    complex_defaults[k] = v
                else:
                    defaults[k] = v

    # 4. Standard Defaults setup
    defaults.setdefault('num_workers', 0)
    defaults.setdefault('bc_loss_scale', 1.0)
    defaults.setdefault('wm_loss_scale', 1.0)
    defaults.setdefault('batch_length', 64)
    defaults.setdefault('batch_size', 16)
    defaults.setdefault('wm_pretrain_steps', 0)
    defaults.setdefault('save_every', 10000)
    defaults.setdefault('robosuite_task', 'Lift')
    defaults.setdefault('robosuite_robots', ['Panda'])
    defaults.setdefault('robosuite_controller', 'OSC_POSE')
    defaults.setdefault('robosuite_reward_shaping', False)
    defaults.setdefault('robosuite_control_freq', 20)
    defaults.setdefault('has_renderer', False)
    defaults.setdefault('has_offscreen_renderer', True)
    defaults.setdefault('use_camera_obs', True)
    defaults.setdefault('camera_depths', False)
    defaults.setdefault('ignore_done', False)
    defaults.setdefault('clip_actions', False)
    defaults.setdefault('bc_sg_wm', False)
    if isinstance(defaults.get("action_mlp"), dict):
        defaults.setdefault("action_mlp_layers", int(defaults["action_mlp"].get("layers", 4)))

    # 5. Build Final Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--env_config", type=str, default=None)

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        if key in complex_defaults:
            continue
            
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    config = parser.parse_args(remaining)

    # 6. Inject Complex Defaults
    for k, v in complex_defaults.items():
        setattr(config, k, v)
    _apply_action_mlp_overrides(config)

    joint_train(config)
