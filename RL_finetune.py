"""RL fine-tuning using a reward model over world-model latents."""

import argparse
import os
import pathlib
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader

import tools
from AC_RL.actor import Actor
from AC_RL.critic import Critic
from AC_RL.RlDataset import RlDataset
from models import WorldModel
from evals.EvalDataset import EvalDataset
from joint_train import collate_episodes, EnvWorker, evaluate_online
from parallel import Parallel
from bc_mlp.BC_MLP_eval import _prepare_obs
from rewards.DITTO import DittoReward
from rewards.Gail import GailReward

import wandb


def _make_batch_iter(dataset, config):
    if dataset.num_episodes == 0:
        raise RuntimeError("RlDataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(getattr(config, "num_workers", 0)),
        collate_fn=collate_episodes,
        drop_last=True,
    )

    while True:
        for batch in loader:
            yield dict(batch)


def _build_reward_model(config, feat_size):
    name = str(getattr(config, "reward_model", "ditto")).lower()
    reward_scale = float(getattr(config, "reward_scale", 1.0))
    reward_shift = float(getattr(config, "reward_shift", 0.0))
    if name in {"ditto", "mse", "max_cos", "cos"}:
        metric = getattr(config, "reward_metric", "max_cos")
        return DittoReward(
            metric=metric,
            reward_scale=reward_scale,
            reward_shift=reward_shift,
        )
    if name == "gail":
        return GailReward(
            input_dim=feat_size,
            hidden_dim=int(getattr(config, "gail_hidden_dim", 256)),
            layers=int(getattr(config, "gail_layers", 2)),
            lr=float(getattr(config, "gail_lr", 1e-4)),
            act=str(getattr(config, "act", "SiLU")),
            use_transitions=bool(getattr(config, "gail_use_transitions", True)),
            reward_scale=reward_scale,
            reward_shift=reward_shift,
        )
    raise ValueError(f"Unknown reward_model: {name}")


def _load_pretrained(world_model, actor, checkpoint_path, device):
    if not checkpoint_path:
        raise FileNotFoundError("checkpoint path is empty.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "wm" in checkpoint:
        world_model.load_state_dict(checkpoint["wm"])
    elif "world_model" in checkpoint:
        world_model.load_state_dict(checkpoint["world_model"])
    else:
        raise KeyError("Checkpoint missing world model weights.")

    policy_state = None
    if "policy" in checkpoint:
        policy_state = checkpoint["policy"]
    elif "actor" in checkpoint:
        policy_state = checkpoint["actor"]
    elif "action_mlp" in checkpoint:
        policy_state = checkpoint["action_mlp"]
    if policy_state is None:
        raise KeyError("Checkpoint missing policy weights.")
    incompatible = actor.actionMLP.load_state_dict(policy_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Policy load_state_dict mismatch:",
            "missing",
            incompatible.missing_keys,
            "unexpected",
            incompatible.unexpected_keys,
        )


def _init_eval_envs(config):
    eval_every = int(getattr(config, "rl_eval_every", getattr(config, "eval_every", 0)))
    eval_episodes = int(getattr(config, "eval_episodes", getattr(config, "eval_episode_num", 0)))
    if eval_every <= 0 or eval_episodes <= 0:
        return None

    required = ("camera_obs_keys", "flip_camera_keys", "bc_cnn_keys_order", "bc_mlp_keys_order")
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        print(f"Skipping online eval; missing config fields: {', '.join(missing)}")
        return None

    crop_h = int(getattr(config, "image_crop_height", 0))
    crop_w = int(getattr(config, "image_crop_width", 0))
    image_hw = (crop_h, crop_w) if (crop_h > 0 and crop_w > 0) else (84, 84)

    num_envs = int(getattr(config, "num_envs", getattr(config, "envs", 1)))
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
        render=getattr(config, "render", False),
    )
    if hasattr(config, "controller_configs"):
        env_config.controller_configs = config.controller_configs

    print(f"Initializing {num_envs} eval envs...")
    envs = []
    for _ in range(num_envs):
        envs.append(Parallel(lambda cfg=env_config, hw=image_hw: EnvWorker(cfg, hw), "process"))
    return envs


def _imagine_policy(wm, actor, start_state, horizon):
    def step(prev, _):
        state, _, _ = prev
        feat = wm.dynamics.get_feat(state)
        action = actor.generate_actions(feat, sample=True)
        succ = wm.dynamics.img_step(state, action)
        return succ, feat, action

    succ, feats, actions = tools.static_scan(
        step, [torch.arange(horizon, device=wm.dynamics._device)], (start_state, None, None)
    )
    feats = feats.permute(1, 0, 2)
    actions = actions.permute(1, 0, 2)
    return feats, actions


def _build_expert_sequence(batch, horizon: int):
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("ditto_verify_horizon must be > 0")

    expert = {}
    for warmup_key, warmup_value in batch.items():
        if not warmup_key.startswith("warmup_"):
            continue
        base_key = warmup_key[len("warmup_") :]
        target_key = f"target_{base_key}"
        if target_key not in batch:
            continue
        if not torch.is_tensor(warmup_value):
            continue
        target_value = batch[target_key]
        if not torch.is_tensor(target_value):
            continue
        warm = warmup_value[:, -1:]
        if horizon > 1:
            tail = target_value[:, : horizon - 1]
            seq = torch.cat([warm, tail], dim=1)
        else:
            seq = warm
        expert[base_key] = seq

    required = ("image", "action", "is_first", "is_terminal")
    missing = [k for k in required if k not in expert]
    if missing:
        raise KeyError(
            "ditto_verify missing required expert keys from EvalDataset batch: "
            + ", ".join(missing)
        )

    t = min(horizon, expert["action"].shape[1], expert["is_terminal"].shape[1])
    for key in list(expert.keys()):
        expert[key] = expert[key][:, :t]

    expert_is_first = torch.zeros_like(expert["is_first"], dtype=torch.float32)
    expert_is_first[:, 0] = 1.0
    expert["is_first"] = expert_is_first
    return expert, t


def ditto_verify(wm, actor, reward_model, config, step, run, eval_envs):
    if eval_envs is None or len(eval_envs) == 0:
        print("Skipping ditto_verify: eval_envs unavailable.")
        return
    if not bool(getattr(config, "robosuite_reward_shaping", False)):
        raise ValueError("ditto_verify requires robosuite_reward_shaping=True.")

    hdf5_path = str(getattr(config, "ditto_verify_hdf5_path", "")).strip()
    eval_dir = str(getattr(config, "ditto_verify_evaldir", "")).strip()
    if not eval_dir:
        eval_dir = str(
            getattr(config, "offline_evaldir", "")
            or getattr(config, "expert_dir", "")
            or getattr(config, "offline_traindir", "")
        ).strip()
    if not hdf5_path:
        raise ValueError("ditto_verify_hdf5_path must be set when ditto_verify=True.")
    if not eval_dir:
        raise ValueError(
            "ditto_verify_evaldir (or offline_evaldir/expert_dir fallback) must be set "
            "when ditto_verify=True."
        )

    horizon = int(getattr(config, "ditto_verify_horizon", 20))
    max_batches = int(getattr(config, "ditto_verify_batches", 1))
    warmup_len = int(getattr(config, "ditto_verify_warmup_len", 5))
    if horizon <= 0 or max_batches <= 0:
        return

    dataset = EvalDataset(
        hdf5_path=hdf5_path,
        eval_dir=eval_dir,
        config=config,
        warmup_len=warmup_len,
        horizon=horizon,
    )
    loader = DataLoader(
        dataset,
        batch_size=len(eval_envs),
        num_workers=0,
        drop_last=True,
    )

    wm_mode = wm.training
    actor_mode = actor.training
    reward_mode = reward_model.training
    wm.eval()
    actor.eval()
    reward_model.eval()

    all_env_curves = []
    all_ditto_curves = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                set_state_promises = [
                    eval_envs[i].set_state(batch["env_state"][i]) for i in range(len(eval_envs))
                ]
                raw_obs_list = [p() for p in set_state_promises]

                current_is_first = [True] * len(eval_envs)
                current_is_terminal = [False] * len(eval_envs)
                prev_action = torch.zeros(
                    (len(eval_envs), config.num_actions), device=config.device
                )
                rssm_state = None

                policy_feat_steps = []
                env_reward_steps = []

                for _ in range(horizon):
                    processed_list = []
                    for i in range(len(eval_envs)):
                        processed_list.append(
                            _prepare_obs(
                                raw_obs_list[i],
                                cnn_keys_order=config.bc_cnn_keys_order,
                                mlp_keys_order=config.bc_mlp_keys_order,
                                camera_keys=config.camera_obs_keys,
                                flip_keys=config.flip_camera_keys,
                                crop_height=config.image_crop_height
                                if config.image_crop_height > 0
                                else None,
                                crop_width=config.image_crop_width
                                if config.image_crop_width > 0
                                else None,
                                is_first=current_is_first[i],
                                is_terminal=current_is_terminal[i],
                                config=config,
                            )
                        )

                    model_obs = {}
                    keys = processed_list[0].keys()
                    for key in keys:
                        if key in ("is_first", "is_terminal"):
                            model_obs[key] = np.asarray(
                                [float(np.asarray(x[key]).reshape(-1)[0]) for x in processed_list],
                                dtype=np.float32,
                            )
                        else:
                            model_obs[key] = np.stack([np.asarray(x[key]) for x in processed_list])

                    model_obs = wm.preprocess(model_obs)
                    embed = wm.encoder(model_obs)
                    post, _ = wm.dynamics.obs_step(
                        rssm_state,
                        prev_action,
                        embed,
                        model_obs["is_first"],
                        sample=False,
                    )
                    rssm_state = post
                    feat = wm.dynamics.get_feat(post)
                    policy_feat_steps.append(feat)

                    action = actor.generate_actions(feat, sample=False)
                    if getattr(config, "clip_actions", False):
                        action = torch.clamp(action, -1.0, 1.0)
                    prev_action = action

                    action_np = action.detach().cpu().numpy()
                    step_promises = [
                        eval_envs[i].step(action_np[i]) for i in range(len(eval_envs))
                    ]
                    step_results = [p() for p in step_promises]

                    rewards = np.asarray([float(result[1]) for result in step_results], dtype=np.float32)
                    dones = [bool(result[2]) for result in step_results]
                    raw_obs_list = [result[0] for result in step_results]
                    env_reward_steps.append(rewards)
                    current_is_first = [False] * len(eval_envs)
                    current_is_terminal = dones

                if not policy_feat_steps or not env_reward_steps:
                    continue

                policy_feats = torch.stack(policy_feat_steps, dim=1)
                env_rewards = np.stack(env_reward_steps, axis=1)

                expert_seq, expert_t = _build_expert_sequence(batch, horizon=horizon)
                t = min(policy_feats.shape[1], env_rewards.shape[1], expert_t)
                if t <= 0:
                    continue
                policy_feats = policy_feats[:, :t]
                for key in list(expert_seq.keys()):
                    expert_seq[key] = expert_seq[key][:, :t]

                expert_data = wm.preprocess(expert_seq)
                expert_embed = wm.encoder(expert_data)
                expert_post, _ = wm.dynamics.observe(
                    expert_embed,
                    expert_data["action"],
                    expert_data["is_first"],
                )
                expert_feats = wm.dynamics.get_feat(expert_post)[:, :t]

                ditto_rewards = reward_model.reward(policy_feats, expert_feats)
                if ditto_rewards.dim() == 1:
                    ditto_rewards = ditto_rewards.unsqueeze(1)
                t = min(t, ditto_rewards.shape[1])
                if t <= 0:
                    continue

                ditto_np = ditto_rewards[:, :t].detach().cpu().numpy()
                env_np = env_rewards[:, :t]
                for i in range(env_np.shape[0]):
                    all_env_curves.append(env_np[i])
                    all_ditto_curves.append(ditto_np[i])

    finally:
        dataset.close()
        if wm_mode:
            wm.train()
        if actor_mode:
            actor.train()
        if reward_mode:
            reward_model.train()

    if not all_env_curves:
        print("ditto_verify: no rollout curves collected.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab20")
    for i, (env_curve, ditto_curve) in enumerate(zip(all_env_curves, all_ditto_curves)):
        color = cmap(i % 20)
        t = np.arange(len(env_curve))
        ax.plot(
            t,
            env_curve,
            color=color,
            linestyle="-",
            alpha=0.70,
            linewidth=1.5,
            label="env_reward" if i == 0 else None,
        )
        ax.plot(
            t,
            ditto_curve,
            color=color,
            linestyle="--",
            alpha=0.85,
            linewidth=1.5,
            label="ditto_reward" if i == 0 else None,
        )
    ax.set_xlabel("time step")
    ax.set_ylabel("reward")
    ax.set_title(f"DITTO Verify Rewards (step={step})")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()

    if run is not None and wandb is not None:
        run.log(
            {"ditto_verify/all_rollouts_reward_vs_time": wandb.Image(fig)},
            commit=False,
        )
    else:
        print("ditto_verify: wandb run not available; skipping plot logging.")
    plt.close(fig)


def rl_finetune(config):

    tools.set_seed_everywhere(int(config.seed))

    if "cuda" not in str(config.device) or not torch.cuda.is_available():
        config.precision = 32

    logdir = pathlib.Path(config.logdir or "logdir/rl_finetune")
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = str(logdir)
    logger = tools.Logger(logdir, 0)

    run = None
    if wandb is not None:
        try:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "RL_finetune"),
                entity=os.getenv("WANDB_ENTITY"),
                name=os.getenv("WANDB_NAME"),
                config=vars(config),
                dir=str(logdir),
                mode=os.getenv("WANDB_MODE", "online"),
            )
            logger.attach_wandb(wandb, run)
        except Exception:
            run = None

    expert_dir = getattr(config, "expert_dir", "") or getattr(
        config, "offline_traindir", ""
    )
    if not expert_dir:
        raise ValueError("expert_dir or offline_traindir must be provided.")

    train_dataset = RlDataset(expert_dir, config, mode="train")
    if train_dataset.num_episodes == 0:
        raise RuntimeError(f"No episodes found in {expert_dir}.")
    episodes = train_dataset.episodes

    if getattr(config, "image_standardize", False) and getattr(
        config, "image_standardize_dataset", False
    ):
        if not hasattr(config, "image_dataset_mean") or not hasattr(
            config, "image_dataset_std"
        ):
            ds_mean, ds_std = tools.compute_image_dataset_stats(episodes)
            ds_std = np.maximum(ds_std, getattr(config, "image_std_min", 1e-3))
            config.image_dataset_mean = ds_mean.astype(np.float32)
            config.image_dataset_std = ds_std.astype(np.float32)

    sample_ep = train_dataset.episode_list[0]
    obs_space, act_space = tools._define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]

    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    feat_size = (
        config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        if config.dyn_discrete
        else config.dyn_stoch + config.dyn_deter
    )
    actor = Actor(config, feat_size, config.num_actions).to(config.device)
    critic = Critic(config, feat_size).to(config.device)

    ckpt_path = getattr(config, "checkpoint", "")
    if not ckpt_path:
        candidate = logdir / "latest.pt"
        if candidate.exists():
            ckpt_path = str(candidate)
    if not ckpt_path:
        raise FileNotFoundError("No checkpoint provided and no latest.pt found.")
    _load_pretrained(wm, actor, ckpt_path, config.device)

    wm.eval()
    wm.requires_grad_(False)

    reward_model = _build_reward_model(config, feat_size).to(config.device)
    reward_model.train()

    batch_iter = _make_batch_iter(train_dataset, config)
    use_amp = bool(
        config.precision == 16
        and torch.cuda.is_available()
        and "cuda" in str(config.device)
    )
    amp_device = "cuda" if use_amp else "cpu"
    horizon_cfg = int(getattr(config, "imag_horizon", 0))

    total_steps = int(config.rl_steps)
    log_every = int(config.rl_log_every)
    save_every = int(config.rl_save_every)
    eval_every = int(config.rl_eval_every)
    eval_envs = _init_eval_envs(config)
    pretrain_steps = int(config.rl_critic_pretrain_steps)
    pretrain_log_every = int(config.rl_critic_pretrain_log_every)

    if pretrain_steps > 0:
        pretrain_step_offset = 0
        actor.requires_grad_(False)
        for pre_step in range(pretrain_steps):
            raw_batch = next(batch_iter)
            data_wm = wm.preprocess(dict(raw_batch))

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=amp_device, enabled=use_amp, dtype=torch.float16
                ):
                    embed = wm.encoder(data_wm)
                    post, _ = wm.dynamics.observe(
                        embed, data_wm["action"], data_wm["is_first"]
                    )
                    expert_feat = wm.dynamics.get_feat(post)
                    start_state = {k: v[:, 0] for k, v in post.items()}
                    horizon = expert_feat.shape[1]
                    if horizon_cfg > 0:
                        horizon = min(horizon, horizon_cfg)
                    expert_feat = expert_feat[:, :horizon]
                    agent_feat, agent_actions = _imagine_policy(
                        wm, actor, start_state, horizon
                    )

            if getattr(config, "gail_use_transitions", True) and horizon < 2:
                raise ValueError("Need at least 2 steps for transition rewards.")

            rewards = reward_model.reward(agent_feat, expert_feat)
            #rewards = reward_model(expert_feat, expert_feat)
            #critic_metrics = critic.update(expert_feat, rewards)
            critic_metrics = critic.update(agent_feat, rewards)

            if pretrain_log_every > 0 and (pre_step % pretrain_log_every == 0):
                logger.step = pretrain_step_offset + pre_step
                metrics = {}
                metrics.update({f"pretrain/{k}": v for k, v in critic_metrics.items()})
                metrics["pretrain/reward_mean"] = float(rewards.mean().item())
                metrics["pretrain/reward_std"] = float(rewards.std().item())
                for name, value in metrics.items():
                    logger.scalar(name, value)
                logger.write(fps=False)

        actor.requires_grad_(True)

    try:
        for step in range(total_steps):
            raw_batch = next(batch_iter)
            data_wm = wm.preprocess(dict(raw_batch))

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=amp_device, enabled=use_amp, dtype=torch.float16
                ):
                    embed = wm.encoder(data_wm)
                    post, _ = wm.dynamics.observe(
                        embed, data_wm["action"], data_wm["is_first"]
                    )
                    expert_feat = wm.dynamics.get_feat(post)
                    start_state = {k: v[:, 0] for k, v in post.items()}
                    horizon = expert_feat.shape[1]
                    if horizon_cfg > 0:
                        horizon = min(horizon, horizon_cfg)
                    expert_feat = expert_feat[:, :horizon]
                    agent_feat, agent_actions = _imagine_policy(
                        wm, actor, start_state, horizon
                    )

            if getattr(config, "gail_use_transitions", True) and horizon < 2:
                raise ValueError("Need at least 2 steps for transition rewards.")

            rewards = reward_model.reward(agent_feat, expert_feat)

            target, weights, value_seq, critic_metrics = critic.update(
                agent_feat, rewards, return_targets=True
            )
            actor_metrics = actor.update(
                agent_feat, agent_actions, target, weights, value_seq
            )

            if log_every > 0 and (step % log_every == 0):
                logger.step = pretrain_steps + step
                metrics = {}
                metrics.update({f"train/{k}": v for k, v in critic_metrics.items()})
                metrics.update({f"train/{k}": v for k, v in actor_metrics.items()})
                metrics["train/reward_mean"] = float(rewards.mean().item())
                metrics["train/reward_std"] = float(rewards.std().item())
                for name, value in metrics.items():
                    logger.scalar(name, value)
                logger.write(fps=False)

            if eval_envs is not None and eval_every > 0 and (step % eval_every == 0):
                print(f"Evaluating online at step {step}...")
                online_metrics = evaluate_online(
                    wm, actor.actionMLP, config, step, run, eval_envs
                )
                if bool(getattr(config, "ditto_verify", False)):
                    ditto_verify(wm, actor, reward_model, config, step, run, eval_envs)
                for name, value in online_metrics.items():
                    logger.scalar(name, value)
                logger.step = pretrain_steps + step
                logger.write(fps=False)
                actor.train()

            if save_every > 0 and (step % save_every == 0):
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "reward_model": reward_model.state_dict(),
                        "step": step,
                        "config": vars(config),
                    },
                    logdir / "rl_latest.pt",
                )
    finally:
        if eval_envs is not None:
            print("Closing eval envs...")
            for env in eval_envs:
                try:
                    env.close()
                except Exception:
                    pass

        if run is not None:
            try:
                wandb.finish()
            except Exception:
                pass


def _parse_config(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--configs", nargs="+")
    pre_parser.add_argument("--env_config", type=str, default=None)
    args, remaining = pre_parser.parse_known_args(argv)

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

    complex_defaults = {}
    if args.env_config:
        env_defaults = configs.get(args.env_config, {})
        if env_defaults:
            for k, v in env_defaults.items():
                if isinstance(v, (dict, list)):
                    complex_defaults[k] = v
                else:
                    defaults[k] = v

    defaults.setdefault("rl_steps", defaults.get("steps", 1e6))
    defaults.setdefault("rl_log_every", defaults.get("log_every", 1e4))
    defaults.setdefault("rl_save_every", defaults.get("save_every", 1e4))
    defaults.setdefault("rl_eval_every", defaults.get("eval_every", 0))
    defaults.setdefault("rl_critic_pretrain_steps", 0)
    defaults.setdefault(
        "rl_critic_pretrain_log_every", defaults.get("rl_log_every", 1e4)
    )
    defaults.setdefault("reward_model", "ditto")
    defaults.setdefault("reward_metric", "max_cos")
    defaults.setdefault("gail_hidden_dim", 256)
    defaults.setdefault("gail_layers", 2)
    defaults.setdefault("gail_lr", 1e-4)
    defaults.setdefault("gail_use_transitions", True)
    defaults.setdefault("eval_episodes", defaults.get("eval_episode_num", 0))
    defaults.setdefault("num_envs", defaults.get("envs", 1))
    defaults.setdefault("max_env_steps", defaults.get("time_limit", 500))
    defaults.setdefault("clip_actions", False)
    defaults.setdefault("checkpoint", "")
    defaults.setdefault("expert_dir", defaults.get("offline_traindir", ""))
    defaults.setdefault("num_workers", 0)
    defaults.setdefault("ditto_verify", False)
    defaults.setdefault("ditto_verify_hdf5_path", "")
    defaults.setdefault("ditto_verify_evaldir", "")
    defaults.setdefault("ditto_verify_warmup_len", 5)
    defaults.setdefault("ditto_verify_horizon", 20)
    defaults.setdefault("ditto_verify_batches", 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--env_config", type=str, default=None)

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        if key in complex_defaults:
            continue
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
    for k, v in complex_defaults.items():
        setattr(config, k, v)
    return config


if __name__ == "__main__":
    config = _parse_config()
    rl_finetune(config)
