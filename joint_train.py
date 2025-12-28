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
from parallel import Parallel
import gym
from collections import defaultdict, OrderedDict

# Ensure the folder containing this script is in python path
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
from models import WorldModel
from networks import MLP
import wandb

# Import Eval helpers
from bc_mlp.BC_MLP_eval import (
    _make_robomimic_env, 
    _prepare_obs, 
    _obs_to_torch, 
    EpisodeVideoRecorder, 
    _extract_success
)

_EXCLUDE_KEYS = {"action", "reward", "discount", "is_first", "is_terminal", "policy_target"}

def to_np(tensor):
    return tensor.detach().cpu().numpy()

# --- Config & Space Utilities ---

def _define_spaces(episode, config):
    """Constructs observation space by filtering episode keys against config regex."""
    obs_spaces = {}
    mlp_pat = config.encoder.get('mlp_keys', '$^')
    cnn_pat = config.encoder.get('cnn_keys', '$^')
    
    for key, value in episode.items():
        if key in _EXCLUDE_KEYS: continue
        
        is_mlp = re.match(mlp_pat, key)
        is_cnn = re.match(cnn_pat, key)
        if not (is_mlp or is_cnn): continue
        
        shape = value.shape[1:]
        if is_cnn:
            h = config.image_crop_height
            w = config.image_crop_width
            if h > 0 and w > 0:
                shape = (h, w, shape[-1])
        
        if value.dtype == np.uint8:
            low, high = 0, 255
            dtype = np.uint8
        else:
            low, high = -np.inf, np.inf
            dtype = np.float32
            
        obs_spaces[key] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
            
    if not obs_spaces:
        raise ValueError("No keys matched config.encoder regex!")
    
    obs_space = gym.spaces.Dict(obs_spaces)
    action = episode.get("action")
    if action is None: raise ValueError("Episode missing 'action'.")
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=action.shape[1:], dtype=np.float32)
    
    return obs_space, act_space

# --- Dataset ---

class JointDataset(Dataset):
    def __init__(self, directory, config, mode='train'):
        self.config = config
        self.mode = mode
        self.directory = pathlib.Path(directory).expanduser()
        self.batch_length = config.batch_length
        self.crop_h = config.image_crop_height
        self.crop_w = config.image_crop_width
        self.do_crop = (self.crop_h > 0 and self.crop_w > 0)
        self.orig_h = 84
        self.orig_w = 84

        if not self.directory.exists():
            # If eval dir doesn't exist, we handle it gracefully later, but here warn
            print(f"Warning: Dataset directory {self.directory} does not exist.")
            self.files = []
            self.episode_list = []
        else:
            self.episodes = tools.load_episodes(self.directory, limit=config.dataset_size)
            self.episode_list = list(self.episodes.values())

        self.num_episodes = len(self.episode_list)
        total_steps = sum(len(ep['action']) for ep in self.episode_list) if self.episode_list else 0
        self.epoch_size = (total_steps // self.batch_length) if self.batch_length else 0
        
        print(f"[{mode}] Loaded {self.num_episodes} episodes, {total_steps} steps.")

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        idx = np.random.randint(0, self.num_episodes)
        return self.episode_list[idx]

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
        collected = defaultdict(list)
        current_len = 0
        
        while current_len < self.batch_length:
            episode = self._sample_episode()
            total_ep_len = len(episode['action'])
            if total_ep_len <= 1: continue 

            needed = self.batch_length - current_len
            valid_pairs = total_ep_len - 1
            if valid_pairs < 1: continue

            if self.mode == 'train':
                start_idx = np.random.randint(0, valid_pairs)
            else:
                # Sequential-ish sampling for eval (or random, doesnt matter much for batch stats)
                # Random is statistically safer for 'epoch' estimates
                start_idx = np.random.randint(0, valid_pairs)

            take = min(needed, valid_pairs - start_idx)
            if take <= 0: continue

            input_slice = slice(start_idx, start_idx + take)
            target_slice = slice(start_idx + 1, start_idx + take + 1)

            # Crops
            raw_imgs = episode['image'][input_slice]
            
            # WM Crop
            t_wm, l_wm = self._get_crop_coords(getattr(self.config, 'wm_random_crop', False))
            collected['image_wm'].append(self._crop(raw_imgs, t_wm, l_wm))

            # BC Crop
            t_bc, l_bc = self._get_crop_coords(getattr(self.config, 'bc_random_crop', False))
            collected['image_bc'].append(self._crop(raw_imgs, t_bc, l_bc))

            # Flags
            first_chunk = episode['is_first'][input_slice].copy()
            if current_len == 0: first_chunk[0] = True # Start of batch
            else: first_chunk[0] = True # Start of appended segment (reset RNN)

            collected['action'].append(episode['action'][input_slice])
            collected['policy_target'].append(episode['action'][target_slice])
            collected['is_first'].append(first_chunk)
            collected['is_terminal'].append(episode['is_terminal'][input_slice])

            # Other Keys (Proprio)
            skip_keys = ['image', 'action', 'is_first', 'is_terminal', 'log_']
            for k, v in episode.items():
                if any(x in k for x in skip_keys): continue
                collected[k].append(v[input_slice])

            current_len += take

        return {k: np.concatenate(v, axis=0) for k, v in collected.items()}

def collate_episodes(batch):
    keys = batch[0].keys()
    res = {}
    for k in keys:
        res[k] = np.stack([b[k] for b in batch], axis=0)
    return res

# --- Evaluation Functions ---

def evaluate_offline(wm, policy, eval_loader, config, step):
    """Computes losses on the evaluation dataset (Batch Mode)."""
    wm.eval()
    policy.eval()
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for i, raw_batch in enumerate(eval_loader):
            if i >= config.offline_eval_batches: break
            
            # 1. WM Prep
            data_wm = raw_batch.copy()
            data_wm['image'] = data_wm.pop('image_wm')
            data_wm = wm.preprocess(data_wm)
            
            # 2. WM Forward
            embed = wm.encoder(data_wm)
            post, prior = wm.dynamics.observe(embed, data_wm['action'], data_wm['is_first'])
            feat = wm.dynamics.get_feat(post)
            
            # 3. WM Losses
            # KL
            kl_loss, kl_value, _, _ = wm.dynamics.kl_loss(
                post, prior, config.kl_free, config.dyn_scale, config.rep_scale
            )
            metrics['eval/kl'].append(kl_value.mean().item())
            
            # Heads (Recon, Reward)
            for name, head in wm.heads.items():
                pred = head(feat)
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        loss = -v.log_prob(data_wm[k])
                        metrics[f'eval/{k}_loss'].append(loss.mean().item())
                else:
                    loss = -pred.log_prob(data_wm[name])
                    metrics[f'eval/{name}_loss'].append(loss.mean().item())

            # 4. BC Forward (Using BC crop)
            img_bc = torch.tensor(raw_batch['image_bc'], device=config.device, dtype=torch.float32) / 255.0
            if config.image_standardize and 'image_mean' in data_wm:
                 img_bc = (img_bc - data_wm['image_mean']) / data_wm['image_std']
            
            # Re-encode for BC to be strictly correct with crop
            data_bc = data_wm.copy()
            data_bc['image'] = img_bc
            embed_bc = wm.encoder(data_bc)
            post_bc, _ = wm.dynamics.observe(embed_bc, data_bc['action'], data_bc['is_first'])
            feat_bc = wm.dynamics.get_feat(post_bc)
            
            # BC Loss
            target = torch.tensor(raw_batch['policy_target'], device=config.device, dtype=torch.float32)
            pred_dist = policy(feat_bc)
            bc_loss = -pred_dist.log_prob(target).mean()
            metrics['eval/bc_loss'].append(bc_loss.item())

    # Aggregate
    agg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return agg_metrics

def evaluate_online(wm, policy, config, step, run):
    """Rolls out policy in parallel Robosuite environments (Closed Loop with RSSM)."""
    wm.eval()
    policy.eval()
    
    # 1. Configuration & Setup
    # Determine image size from config (handle crop vs no-crop)
    image_hw = (config.image_crop_height, config.image_crop_width)
    if image_hw[0] == 0: image_hw = (84, 84)

    num_envs = config.num_envs if hasattr(config, 'num_envs') else 1
    total_episodes = config.eval_episodes
    video_dir = pathlib.Path(config.logdir) / "eval_videos" / f"step_{step}"
    
    # Helper class to create envs in subprocesses (from BC_Sweep.py)
    class _EnvWorker:
        def __init__(self, cfg, image_size):
            self._cfg = cfg
            self._image_size = image_size
            self._env = None

        def _ensure_env(self):
            if self._env is None:
                self._env = _make_robomimic_env(self._cfg, self._image_size)

        def reset(self, seed=None):
            self._ensure_env()
            if seed is not None:
                try:
                    return self._env.reset(seed=seed)
                except TypeError:
                    self._env.seed(seed)
                    return self._env.reset()
            return self._env.reset()

        def step(self, action):
            self._ensure_env()
            obs, reward, done, info = self._env.step(action)
            return obs, reward, done, info, _extract_success(info, self._env)

        def close(self):
            if self._env: self._env.close()

    # 2. Initialize Parallel Environments
    print(f"Starting Online Eval ({total_episodes} episodes) with {num_envs} parallel envs...")
    envs = []
    for i in range(num_envs):
        # We pass a copy of config to avoid pickling issues or shared state modification
        worker = _EnvWorker(config, image_hw)
        envs.append(Parallel(worker, "process"))

    # 3. State Tracking
    # We maintain persistent state for the batch of environments
    obs_batch = [None] * num_envs
    
    # RSSM State (Batch, State_Dim) - initialized to None (reset)
    rssm_state = None 
    # Previous Action (Batch, Act_Dim)
    prev_action = torch.zeros((num_envs, config.num_actions), device=config.device)
    # Is First Flag (Batch, 1)
    is_first = torch.ones((num_envs, 1), device=config.device)
    
    # Episode tracking
    episode_rewards = [0.0] * num_envs
    episode_steps = [0] * num_envs
    episode_success = [False] * num_envs
    completed_episodes = 0
    
    # Results
    final_rewards = []
    final_successes = []
    
    # Video Recorders
    recorders = [
        EpisodeVideoRecorder(
            directory=video_dir,
            fps=20,
            camera_key=config.camera_obs_keys[0],
            camera_keys=config.camera_obs_keys,
            flip_keys=config.flip_camera_keys
        ) for _ in range(num_envs)
    ]

    # 4. Initial Reset
    # We assign a global episode ID to each slot to track total progress
    # global_ep_idx tracks the next available episode ID to assign
    global_ep_idx = 0
    env_episode_ids = [0] * num_envs # ID of the episode currently running in env i

    for i in range(num_envs):
        if global_ep_idx < total_episodes:
            obs_batch[i] = envs[i].reset()() # Blocking reset
            recorders[i].start_episode(global_ep_idx)
            recorders[i].add_frame(obs_batch[i])
            env_episode_ids[i] = global_ep_idx
            global_ep_idx += 1
        else:
            # If we have more envs than episodes (unlikely but possible), idle them
            obs_batch[i] = None

    # 5. Rollout Loop
    while completed_episodes < total_episodes:
        # A. Filter active environments
        active_indices = [i for i, obs in enumerate(obs_batch) if obs is not None]
        if not active_indices: break

        # B. Prepare Batch Data
        # We need to construct a batch containing ONLY the active indices to pass to the model,
        # OR we pass the full batch and mask operations. 
        # Standard Dreamer approach: Pass full batch, but only update active slots.
        # However, since 'rssm_state' is a complex dict/tuple, slicing it is tricky without helpers.
        # We will assume we run all 'num_envs' and just ignore outputs for idle ones, 
        # OR we reconstruct the batch every step. 
        # Reconstructing is safer for 'rssm_state' correctness if indices shift.
        # BUT 'rssm_state' relies on temporal continuity. Index 'i' must always correspond to env 'i'.
        # So we MUST run the full batch size 'num_envs' through the model every step, 
        # even if some envs are idle (we can just feed zeros/dummy data for idle ones).
        
        # 1. Process Observations
        batch_obs_dict = defaultdict(list)
        
        for i in range(num_envs):
            if obs_batch[i] is None:
                # Dummy observation for idle envs (zeros)
                # We need valid shapes. Let's use the shape from a valid one or config
                # Fallback shapes from config if all are None (loop would break anyway)
                # Just reuse the last valid obs or zeros.
                # Simplest: Just use zeros matching expected shapes
                pass # Handled below by _obs_to_torch padding logic if we wanted, 
                     # but actually let's just make sure we don't crash.
            else:
                # Prepare single obs
                processed = _prepare_obs(
                    obs_batch[i],
                    cnn_keys_order=config.bc_cnn_keys_order,
                    mlp_keys_order=config.bc_mlp_keys_order,
                    camera_keys=config.camera_obs_keys,
                    flip_keys=config.flip_camera_keys,
                    crop_height=config.image_crop_height if config.image_crop_height > 0 else None,
                    crop_width=config.image_crop_width if config.image_crop_width > 0 else None
                )
                for k, v in processed.items():
                    batch_obs_dict[k].append(v)

        # If an env is idle, we must still provide data to maintain batch size for RSSM
        # We duplicate the first valid obs for idle slots to keep shapes valid
        first_valid_idx = active_indices[0]
        for k in list(batch_obs_dict.keys()): # get keys from the valid entries
            # We have len(active_indices) items. We need num_envs items.
            # This logic is tricky. Let's build the list fully initialized.
            pass
        
        # Better approach: Build list of size num_envs directly
        full_batch_lists = defaultdict(list)
        valid_template = _prepare_obs(obs_batch[first_valid_idx], **{
            'cnn_keys_order': config.bc_cnn_keys_order, 
            'mlp_keys_order': config.bc_mlp_keys_order,
            'camera_keys': config.camera_obs_keys,
            'flip_keys': config.flip_camera_keys,
            'crop_height': config.image_crop_height if config.image_crop_height > 0 else None,
            'crop_width': config.image_crop_width if config.image_crop_width > 0 else None
        }) # Template for shapes

        for i in range(num_envs):
            if obs_batch[i] is None:
                # Use zero arrays matching template shapes
                for k, v in valid_template.items():
                    full_batch_lists[k].append(np.zeros_like(v))
            else:
                processed = _prepare_obs(
                    obs_batch[i],
                    cnn_keys_order=config.bc_cnn_keys_order,
                    mlp_keys_order=config.bc_mlp_keys_order,
                    camera_keys=config.camera_obs_keys,
                    flip_keys=config.flip_camera_keys,
                    crop_height=config.image_crop_height if config.image_crop_height > 0 else None,
                    crop_width=config.image_crop_width if config.image_crop_width > 0 else None
                )
                for k, v in processed.items():
                    full_batch_lists[k].append(v)

        # Convert to Tensor Batch
        data = {}
        for k, v_list in full_batch_lists.items():
            # stack -> (Batch, ...)
            arr = np.stack(v_list)
            tensor = torch.as_tensor(arr, device=config.device).float()
            if k == 'image':
                tensor = tensor / 255.0
                if config.image_standardize and wm._dataset_image_mean is not None:
                    tensor = (tensor - wm._dataset_image_mean) / wm._dataset_image_std
            data[k] = tensor

        # C. Forward Model
        with torch.no_grad():
            # Encoder
            embed = wm.encoder(data) # (B, E)
            
            # RSSM Step
            # obs_step handles the temporal step for the batch
            # prev_action, is_first must match batch size
            post, _ = wm.dynamics.obs_step(
                rssm_state, prev_action, embed, is_first
            )
            rssm_state = post # Update state for next step
            
            # Policy
            feat = wm.dynamics.get_feat(post)
            pred_dist = policy(feat)
            if config.actor['dist'] == 'onehot':
                action_tensor = pred_dist.mode()
            else:
                action_tensor = pred_dist.mode()
            
            # Update prev_action for next step
            prev_action = action_tensor

        # D. Step Environment
        action_np = action_tensor.cpu().numpy()
        if config.clip_actions:
            action_np = np.clip(action_np, -1.0, 1.0)

        # Dispatch async steps
        promises = []
        for i in range(num_envs):
            if obs_batch[i] is not None:
                promises.append(envs[i].step(action_np[i]))
            else:
                promises.append(None)

        # Collect results
        for i, promise in enumerate(promises):
            if promise is None: continue
            
            obs, reward, done, info, success = promise()
            
            # Track
            episode_rewards[i] += reward
            episode_steps[i] += 1
            episode_success[i] = (episode_success[i] or success)
            
            # Update recorder
            recorders[i].add_frame(obs)
            
            # Check Termination
            env_done = done or (episode_steps[i] >= config.max_env_steps) or success
            
            if env_done:
                # Episode Finished
                completed_episodes += 1
                final_rewards.append(episode_rewards[i])
                final_successes.append(episode_success[i])
                
                # Save video (if first few episodes)
                vid_path = recorders[i].finish_episode()
                if vid_path and env_episode_ids[i] == 0:
                    run.log({"eval_online/video": wandb.Video(str(vid_path), fps=20, format="mp4")}, commit=False)
                
                # Reset for next episode if available
                if global_ep_idx < total_episodes:
                    obs_batch[i] = envs[i].reset()()
                    recorders[i].start_episode(global_ep_idx)
                    recorders[i].add_frame(obs_batch[i])
                    
                    # Reset Tracking
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    episode_success[i] = False
                    env_episode_ids[i] = global_ep_idx
                    global_ep_idx += 1
                    
                    # Reset RSSM logic for this slot
                    is_first[i] = 1.0
                    prev_action[i] = 0.0
                    # Note: We don't manually reset rssm_state[i] because obs_step handles 
                    # state reset internally when is_first=1 is passed.
                else:
                    # No more episodes, mark idle
                    obs_batch[i] = None
                    is_first[i] = 0.0 # Don't care
            else:
                # Continue Episode
                obs_batch[i] = obs
                is_first[i] = 0.0 # Next step is not first

    # Cleanup
    for env in envs:
        try: env.close()
        except: pass

    # Metrics
    metrics = {
        "eval_online/success_rate": np.mean(final_successes) if final_successes else 0.0,
        "eval_online/mean_return": np.mean(final_rewards) if final_rewards else 0.0
    }
    return metrics
# --- Main ---

def joint_train(config):
    tools.set_seed_everywhere(config.seed)
    
    # 1. WandB & Dirs
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

    # 2. Data
    train_dataset = JointDataset(config.offline_traindir, config, mode='train')
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers, collate_fn=collate_episodes, 
        pin_memory=True, drop_last=True
    )
    
    eval_dataset = JointDataset(config.offline_evaldir, config, mode='eval')
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=0, collate_fn=collate_episodes, pin_memory=True
    )
    
    train_iter = iter(train_loader) 
    # Helper to cycle infinite
    def cycle(loader):
        while True:
            for b in loader: yield b
    train_iter = cycle(train_loader)

    # 3. Model Init
    print("Inferring Observation Space...")
    sample_ep = train_dataset.episode_list[0]
    obs_space, act_space = _define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]
    
    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    
    # Policy (MLP on RSSM Features)
    if config.dyn_discrete:
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
        feat_size = config.dyn_stoch + config.dyn_deter
        
    policy = MLP(
        inp_dim=feat_size,
        shape=(config.num_actions,),
        layers=4, units=1024, act=config.act, norm=config.norm,
        dist='mse', outscale=1.0, device=config.device, name='Policy'
    ).to(config.device)

    optimizer = torch.optim.Adam(
        list(wm.parameters()) + list(policy.parameters()), 
        lr=config.model_lr, weight_decay=config.weight_decay
    )

    # 4. Training Loop
    print("Starting Joint Training...")
    step = 0
    
    for _ in range(int(config.steps)):
        wm.train()
        policy.train()
        
        # --- Data Prep ---
        raw_batch = next(train_iter)
        
        # WM Data
        data_wm = raw_batch.copy()
        data_wm['image'] = data_wm.pop('image_wm')
        data_wm = wm.preprocess(data_wm)
        
        # BC Data
        img_bc = torch.tensor(raw_batch['image_bc'], device=config.device, dtype=torch.float32) / 255.0
        if config.image_standardize and 'image_mean' in data_wm:
             img_bc = (img_bc - data_wm['image_mean']) / data_wm['image_std']
        
        # --- 1. WM Branch ---
        embed_wm = wm.encoder(data_wm) 
        post_wm, prior_wm = wm.dynamics.observe(embed_wm, data_wm['action'], data_wm['is_first'])
        kl_loss, kl_val, _, _ = wm.dynamics.kl_loss(
            post_wm, prior_wm, config.kl_free, config.dyn_scale, config.rep_scale
        )
        
        feat_wm = wm.dynamics.get_feat(post_wm)
        recon_losses = 0
        for name, head in wm.heads.items():
            pred = head(feat_wm)
            if isinstance(pred, dict):
                for k, v in pred.items(): recon_losses -= v.log_prob(data_wm[k])
            else:
                recon_losses -= pred.log_prob(data_wm[name])
        
        wm_loss = recon_losses.mean() + kl_loss
        
        # --- 2. BC Branch ---
        data_bc = data_wm.copy()
        data_bc['image'] = img_bc
        embed_bc = wm.encoder(data_bc)
        post_bc, _ = wm.dynamics.observe(embed_bc, data_bc['action'], data_bc['is_first'])
        feat_bc = wm.dynamics.get_feat(post_bc)
        
        pred_dist = policy(feat_bc)
        target = torch.tensor(raw_batch['policy_target'], device=config.device, dtype=torch.float32)
        bc_loss = -pred_dist.log_prob(target).mean()
        
        # Optimization
        total_loss = (config.wm_loss_scale * wm_loss) + (config.bc_loss_scale * bc_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(wm.parameters()) + list(policy.parameters()), config.grad_clip)
        optimizer.step()
        
        step += 1
        
        # --- Logging ---
        if step % config.log_every == 0:
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/wm_loss": wm_loss.item(),
                "train/bc_loss": bc_loss.item(),
                "train/kl": kl_val.mean().item()
            }, step=step)
            
        # --- Evaluation ---
        if step % config.eval_every == 0:
            print(f"Evaluating at step {step}...")
            
            # Offline
            off_metrics = evaluate_offline(wm, policy, eval_loader, config, step)
            wandb.log(off_metrics, step=step)
            
            # Online
            on_metrics = evaluate_online(wm, policy, config, step, run)
            wandb.log(on_metrics, step=step)
            
            # Checkpoint
            torch.save({
                'wm': wm.state_dict(),
                'policy': policy.state_dict()
            }, logdir / "latest.pt")

    print("Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())
    
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
        
    # Inject Defaults
    defaults.setdefault('num_workers', 0)
    defaults.setdefault('bc_loss_scale', 1.0)
    defaults.setdefault('wm_loss_scale', 1.0)
    defaults.setdefault('batch_length', 64)
    defaults.setdefault('batch_size', 16)
    defaults.setdefault('image_crop_height', 0)
    defaults.setdefault('image_crop_width', 0)
    defaults.setdefault('wm_random_crop', True)
    defaults.setdefault('bc_random_crop', True)
    
    # Eval Defaults (robosuite)
    defaults.setdefault('eval_episodes', 5)
    defaults.setdefault('max_env_steps', 500)
    defaults.setdefault('offline_eval_batches', 10)
    
    # Fill in required list keys if missing (for _prepare_obs)
    defaults.setdefault('bc_cnn_keys_order', ['image'])
    defaults.setdefault('bc_mlp_keys_order', [
        "robot0_joint_pos", "robot0_joint_vel", "robot0_gripper_qpos", "robot0_gripper_qvel",
        "aux_robot0_joint_pos_sin", "aux_robot0_joint_pos_cos"
    ])
    defaults.setdefault('camera_obs_keys', ['agentview_image', 'robot0_eye_in_hand_image'])
    defaults.setdefault('flip_camera_keys', ['agentview_image', 'robot0_eye_in_hand_image'])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    config = parser.parse_args(remaining)
    joint_train(config)