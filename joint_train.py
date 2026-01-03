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
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import gym
from collections import defaultdict, OrderedDict
from types import SimpleNamespace

# Ensure the folder containing this script is in python path
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
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
try:
    from BC_Sweep import _load_env_block
except ImportError:
    _load_env_block = None
    print("Warning: Could not import _load_env_block from BC_Sweep.")

_EXCLUDE_KEYS = {"action", "reward", "discount", "is_first", "is_terminal", "policy_target"}

def to_np(tensor):
    return tensor.detach().cpu().numpy()

# --- Custom Action MLP (Raw Output) ---
class ActionMLP(nn.Module):
    def __init__(self, inp_dim, shape, layers=4, units=1024, act="SiLU", norm=False):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        out_dim = shape[0]
        
        net_layers = []
        last_dim = inp_dim
        
        for _ in range(layers):
            net_layers.append(nn.Linear(last_dim, units, bias=False))
            if norm:
                net_layers.append(nn.LayerNorm(units, eps=1e-03))
            net_layers.append(act_cls())
            last_dim = units
            
        net_layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*net_layers)
        
        # Weight Init
        self.apply(tools.weight_init)

    def forward(self, features):
        return self.net(features)

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
            print(f"Warning: Dataset directory {self.directory} does not exist.")
            self.episodes = {}
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

            start_idx = np.random.randint(0, valid_pairs)
            take = min(needed, valid_pairs - start_idx)
            if take <= 0: continue

            input_slice = slice(start_idx, start_idx + take)
            target_slice = slice(start_idx + 1, start_idx + take + 1)

            raw_imgs = episode['image'][input_slice]
            
            t_wm, l_wm = self._get_crop_coords(getattr(self.config, 'wm_random_crop', False))
            collected['image_wm'].append(self._crop(raw_imgs, t_wm, l_wm))

            t_bc, l_bc = self._get_crop_coords(getattr(self.config, 'bc_random_crop', False))
            collected['image_bc'].append(self._crop(raw_imgs, t_bc, l_bc))

            first_chunk = episode['is_first'][input_slice].copy()
            if current_len == 0: first_chunk[0] = True 
            else: first_chunk[0] = True 

            collected['action'].append(episode['action'][input_slice])
            collected['policy_target'].append(episode['action'][target_slice])
            collected['is_first'].append(first_chunk)
            collected['is_terminal'].append(episode['is_terminal'][input_slice])

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
    wm.eval()
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
            metrics['eval/kl'].append(kl_value.mean().item())
            
            for name, head in wm.heads.items():
                pred = head(feat)
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        loss = -v.log_prob(data_wm[k])
                        metrics[f'eval/{k}_loss'].append(loss.mean().item())
                else:
                    loss = -pred.log_prob(data_wm[name])
                    metrics[f'eval/{name}_loss'].append(loss.mean().item())

            img_bc = torch.tensor(raw_batch['image_bc'], device=config.device, dtype=torch.float32) / 255.0
            if config.image_standardize and 'image_mean' in data_wm:
                 img_bc = (img_bc - data_wm['image_mean']) / data_wm['image_std']
            
            data_bc = data_wm.copy()
            data_bc['image'] = img_bc
            embed_bc = wm.encoder(data_bc)
            post_bc, _ = wm.dynamics.observe(embed_bc, data_bc['action'], data_bc['is_first'])
            feat_bc = wm.dynamics.get_feat(post_bc)
            
            # --- BC Loss (MSE) ---
            target = torch.tensor(raw_batch['policy_target'], device=config.device, dtype=torch.float32)
            pred_action = policy(feat_bc)
            bc_loss = F.mse_loss(pred_action, target)
            metrics['eval/bc_loss'].append(bc_loss.item())

    agg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return agg_metrics

# --- GLOBAL WORKER CLASS (Moved outside evaluate_online to support pickling/reuse) ---
class EnvWorker:
    def __init__(self, env_cfg, img_size):
        self._cfg = env_cfg
        self._image_size = img_size
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

def evaluate_online(wm, policy, config, step, run, envs):
    """
    Evaluates policy online using persistent environments 'envs'.
    """
    wm.eval()
    policy.eval()
    
    num_envs = len(envs)
    total_episodes = config.eval_episodes
    
    # Handle optional video recording (skip if logdir is None)
    video_dir = None
    if config.logdir is not None:
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
    
    # Initial Reset
    print(f"Starting Online Eval ({total_episodes} episodes)...")
    for i in range(num_envs):
        if global_ep_idx < total_episodes:
            obs_batch[i] = envs[i].reset()() 
            if recorders[i]:
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
            crop_width=config.image_crop_width if config.image_crop_width > 0 else None
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
                    crop_width=config.image_crop_width if config.image_crop_width > 0 else None
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
            if recorders[i]:
                recorders[i].add_frame(obs)
            
            env_done = done or (episode_steps[i] >= getattr(config, "max_env_steps", 500)) or success
            
            if env_done:
                completed_episodes += 1
                final_rewards.append(episode_rewards[i])
                final_successes.append(episode_success[i])
                
                vid_path = recorders[i].finish_episode() if recorders[i] else None
                if vid_path and env_episode_ids[i] < 3:
                    if run: 
                        run.log({f"eval_online/video_{env_episode_ids[i]}": wandb.Video(str(vid_path), fps=20, format="mp4")}, commit=False)
                
                if global_ep_idx < total_episodes:
                    obs_batch[i] = envs[i].reset()()
                    if recorders[i]:
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
        "eval_online/mean_return": np.mean(final_rewards) if final_rewards else 0.0
    }
    return metrics

# --- Main ---

def joint_train(config):
    tools.set_seed_everywhere(config.seed)
    
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
    
    def cycle(loader):
        while True:
            for b in loader: yield b
    train_iter = cycle(train_loader)

    print("Inferring Observation Space...")
    sample_ep = train_dataset.episode_list[0]
    obs_space, act_space = _define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]
    
    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    
    # --- Initialize ActionMLP (Raw Policy) ---
    if config.dyn_discrete:
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
        feat_size = config.dyn_stoch + config.dyn_deter
        
    policy = ActionMLP(
        inp_dim=feat_size,
        shape=(config.num_actions,),
        layers=4,
        units=1024, act=config.act, norm=config.norm
    ).to(config.device)
    
    # Print Param Count for Sanity Check
    print(f"--- Parameter Check ---")
    print(f"Policy Params:      {sum(p.numel() for p in policy.parameters()):,}")
    print(f"World Model Params: {sum(p.numel() for p in wm.parameters()):,}")
    print(f"-----------------------")

    optimizer = torch.optim.Adam(
        list(wm.parameters()) + list(policy.parameters()), 
        lr=config.model_lr, weight_decay=config.weight_decay
    )

    # --- SETUP EVAL ENVS ONCE ---
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

    eval_envs = []
    try:
        # Create envs once
        for _ in range(config.num_envs):
            worker = EnvWorker(env_config, image_hw)
            eval_envs.append(Parallel(worker, "process"))

        print("Starting Joint Training...")
        step = 0
        
        for _ in range(int(config.steps)):
            wm.train()
            policy.train()
            
            raw_batch = next(train_iter)
            
            # 1. WM Branch
            data_wm = raw_batch.copy()
            data_wm['image'] = data_wm.pop('image_wm')
            data_wm = wm.preprocess(data_wm)
            
            # 2. BC Data Preparation
            img_bc = torch.tensor(raw_batch['image_bc'], device=config.device, dtype=torch.float32) / 255.0
            if config.image_standardize and 'image_mean' in data_wm:
                 img_bc = (img_bc - data_wm['image_mean']) / data_wm['image_std']
            
            # 3. Forward World Model
            embed_wm = wm.encoder(data_wm) 
            post_wm, prior_wm = wm.dynamics.observe(embed_wm, data_wm['action'], data_wm['is_first'])
            
            # Calculate KL Loss
            kl_loss, kl_val, _, _ = wm.dynamics.kl_loss(
                post_wm, prior_wm, config.kl_free, config.dyn_scale, config.rep_scale
            )
            
            # Calculate Reconstruction Losses
            feat_wm = wm.dynamics.get_feat(post_wm)
            recon_losses = 0
            for name, head in wm.heads.items():
                pred = head(feat_wm)
                if isinstance(pred, dict):
                    for k, v in pred.items(): 
                        recon_losses -= v.log_prob(data_wm[k])
                else:
                    recon_losses -= pred.log_prob(data_wm[name])
            
            wm_loss = (recon_losses + kl_loss).mean()
            
            # 4. Forward BC Branch
            data_bc = data_wm.copy()
            data_bc['image'] = img_bc
            embed_bc = wm.encoder(data_bc)
            post_bc, _ = wm.dynamics.observe(embed_bc, data_bc['action'], data_bc['is_first'])
            feat_bc = wm.dynamics.get_feat(post_bc)
            
            # --- BC Loss (Raw MSE) ---
            target = torch.tensor(raw_batch['policy_target'], device=config.device, dtype=torch.float32)
            pred_action = policy(feat_bc)
            bc_loss = F.mse_loss(pred_action, target)
            
            # 5. Optimization
            total_loss = (config.wm_loss_scale * wm_loss) + (config.bc_loss_scale * bc_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(wm.parameters()) + list(policy.parameters()), config.grad_clip)
            optimizer.step()
            
            step += 1
            
            if step % config.log_every == 0:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/wm_loss": wm_loss.item(),
                    "train/bc_loss": bc_loss.item(),
                    "train/kl": kl_val.mean().item()
                }, step=step)
                
            if step % config.eval_every == 0:
                print(f"Evaluating at step {step}...")
                off_metrics = evaluate_offline(wm, policy, eval_loader, config, step)
                wandb.log(off_metrics, step=step)
                
                # Pass reused envs here
                on_metrics = evaluate_online(wm, policy, config, step, run, eval_envs)
                wandb.log(on_metrics, step=step)
                
                torch.save({
                    'wm': wm.state_dict(),
                    'policy': policy.state_dict()
                }, logdir / "latest.pt")

            if step % config.save_every == 0:
                torch.save({
                    'wm': wm.state_dict(),
                    'policy': policy.state_dict(),
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, logdir / f"step_{step}.pt")

        print("Training Finished.")
    
    finally:
        print("Closing Eval Envs...")
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

    joint_train(config)