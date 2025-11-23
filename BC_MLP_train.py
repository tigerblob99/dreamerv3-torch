import argparse
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault("MUJOCO_GL", "osmesa")

import ruamel.yaml as yaml

import torch
import wandb

import networks
import tools
from offline_train import _infer_spaces


def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _load_config(config_names, remaining_args):
    configs_path = pathlib.Path(__file__).with_name("configs.yaml")
    configs = yaml.safe_load(configs_path.read_text())
    config = {}
    name_list = ["defaults", *(config_names or [])]
    for name in name_list:
        _recursive_update(config, configs[name])

    # Ensure BC defaults exist even if a config block omits them.
    config.setdefault("bc_epochs", 10)
    config.setdefault("bc_batch_size", 64)
    config.setdefault("bc_lr", 3e-4)
    config.setdefault("bc_weight_decay", 0.0)
    config.setdefault("bc_eval_every", 1)
    config.setdefault("bc_eval_split", 0.1)
    config.setdefault("bc_max_steps", -1)
    config.setdefault("bc_grad_clip", 100.0)
    config.setdefault("bc_shuffle", True)
    config.setdefault("bc_loss", "mse")
    config.setdefault("bc_save_path", "")

    parser = argparse.ArgumentParser()
    for key, value in sorted(config.items()):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=value)
    parser.add_argument("--checkpoint", type=str, default=config.get("checkpoint", ""))
    return parser.parse_args(remaining_args)


def _build_encoder(config, obs_space):
    shapes = {key: tuple(space.shape) for key, space in obs_space.spaces.items()}
    encoder = networks.MultiEncoder(shapes, **config.encoder).to(config.device)
    encoder.eval()
    return encoder


def _extract_encoder_state(agent_state_dict):
    prefix = "_wm._orig_mod.encoder."
    filtered = {key[len(prefix) :]: value for key, value in agent_state_dict.items() if key.startswith(prefix)}
    if not filtered:
        raise KeyError("Encoder weights not found in checkpoint.")
    return filtered


def BC_MLP(config):
    if not config.offline_traindir:
        raise ValueError("--offline_traindir must be provided to infer observation shapes.")
    tools.set_seed_everywhere(config.seed)

    dataset = tools.load_episodes(config.offline_traindir, limit=1)
    if not dataset:
        raise RuntimeError(f"No episodes found in {config.offline_traindir}.")
    first_episode = next(iter(dataset.values()))
    obs_space, act_space = _infer_spaces(first_episode)

    encoder = _build_encoder(config, obs_space)

    def build_action_mlp(encoder_module, action_space, hidden_units=1024, hidden_layers=4, act_name="SiLU", norm=False, device=None):
        """Builds a raw-output MLP with hidden activations and final activation sized to action space.

        The network returns non-distribution raw activations (no sampling wrapper).
        Architecture: [encoder_outdim -> 1024 x (hidden_layers) -> action_dim] with activation after final layer.
        """
        act_cls = getattr(torch.nn, act_name)
        action_dim = action_space.shape[0]
        layers = []
        inp_dim = getattr(encoder_module, "outdim", None)
        if inp_dim is None:
            raise AttributeError("Encoder module missing 'outdim' attribute needed for BC MLP input size.")
        for i in range(hidden_layers):
            layers.append(torch.nn.Linear(inp_dim, hidden_units, bias=False))
            if norm:
                layers.append(torch.nn.LayerNorm(hidden_units, eps=1e-03))
            layers.append(act_cls())
            inp_dim = hidden_units
        layers.append(torch.nn.Linear(inp_dim, action_dim))  # no final activation (raw action outputs)
        mlp = torch.nn.Sequential(*layers)
        mlp.to(device or config.device)
        # weight init similar to other modules
        for m in mlp.modules():
            if isinstance(m, torch.nn.Linear):
                tools.weight_init(m)
        return mlp

    action_mlp = build_action_mlp(encoder, act_space)

    checkpoint = config.checkpoint or (config.logdir and pathlib.Path(config.logdir) / "latest.pt")
    checkpoint_path = pathlib.Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    agent_state = ckpt.get("agent_state_dict")
    if agent_state is None:
        raise KeyError("agent_state_dict missing from checkpoint.")

    encoder_state = _extract_encoder_state(agent_state)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded encoder from {checkpoint_path}.")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    print(f"Built action MLP: hidden_layers=4, hidden_units=1024, action_dim={act_space.shape[0]}")
    return encoder, action_mlp

def _flatten_episodes(episodes):
    """Flatten episodes dict into list of (obs_dict, action) per timestep."""
    samples = []
    for ep in episodes.values():
        length = ep['action'].shape[0]
        for t in range(length):
            obs = {k: v[t] for k, v in ep.items() if k not in ('action',) and not k.startswith('log_')}
            action = ep['action'][t]
            samples.append((obs, action))
    return samples


def _sample_batch(samples, batch_size, device):
    idxs = np.random.randint(0, len(samples), size=batch_size)
    batch_obs = {}
    batch_actions = []
    for i in idxs:
        obs, act = samples[i]
        for k, v in obs.items():
            batch_obs.setdefault(k, []).append(v)
        batch_actions.append(act)
    # stack
    for k in batch_obs:
        arr = np.asarray(batch_obs[k])
        if arr.ndim == 1:
            arr = arr[:, None]
        batch_obs[k] = torch.as_tensor(arr, device=device).float()
    batch_actions = torch.as_tensor(np.asarray(batch_actions), device=device).float()
    return batch_obs, batch_actions


def BC_MLP_train(config, encoder, action_mlp):
    print("Preparing BC dataset...")
    train_eps = tools.load_episodes(config.offline_traindir, limit=config.dataset_size)
    eval_dir = getattr(config, "offline_evaldir", None)
    eval_eps = tools.load_episodes(eval_dir, limit=config.dataset_size) if eval_dir else {}
    if not train_eps:
        raise RuntimeError("No episodes for BC training.")
    episode_items = list(train_eps.items())
    if config.bc_eval_split > 0:
        eval_cut = int(len(episode_items) * config.bc_eval_split)
        eval_eps = dict(episode_items[:eval_cut]) if eval_cut > 0 else {}
        train_eps = dict(episode_items[eval_cut:])
    print(f"BC train episodes: {len(train_eps)}, eval episodes: {len(eval_eps)}")

    train_samples = _flatten_episodes(train_eps)
    eval_samples = _flatten_episodes(eval_eps) if eval_eps else []
    print(f"Flattened train samples: {len(train_samples)}, eval samples: {len(eval_samples)}")

    encoder.requires_grad_(False)
    action_mlp.train()

    # Optimizer
    optimizer = torch.optim.Adam(action_mlp.parameters(), lr=config.bc_lr, weight_decay=config.bc_weight_decay)
    loss_fn = torch.nn.MSELoss() if config.bc_loss == 'mse' else torch.nn.L1Loss()

    steps_per_epoch = max(1, len(train_samples) // config.bc_batch_size)
    total_steps = config.bc_epochs * steps_per_epoch if config.bc_max_steps < 0 else config.bc_max_steps
    print(f"BC training: epochs={config.bc_epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

    run_name = f"bc_epochs{config.bc_epochs}_lr{config.bc_lr:g}"
    run = wandb.init(
        entity="4yp-a2i",
        project="BC_practice",
        name=run_name,
        config=vars(config),
    )
    wandb.watch(action_mlp, log="all")
    def evaluate():
        if not eval_samples:
            return None
        action_mlp.eval()
        losses = []
        with torch.no_grad():
            batch_size = min(config.bc_batch_size, len(eval_samples))
            # iterate in chunks
            for start in range(0, len(eval_samples), batch_size):
                chunk = eval_samples[start:start+batch_size]
                if not chunk:
                    continue
                obs_list = {k: [] for k in chunk[0][0].keys()}
                act_list = []
                for obs, act in chunk:
                    for k,v in obs.items():
                        obs_list[k].append(v)
                    act_list.append(act)
                for k in obs_list:
                    arr = np.asarray(obs_list[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    obs_list[k] = torch.as_tensor(arr, device=config.device).float()
                acts = torch.as_tensor(np.asarray(act_list), device=config.device).float()
                with torch.no_grad():
                    enc_in = {k: v for k,v in obs_list.items()}
                    embedding = encoder(enc_in)  # (batch, embed)
                    pred = action_mlp(embedding)
                    loss = loss_fn(pred, acts)
                losses.append(loss.item())

        # Visualization of the first 100 steps
        if len(eval_samples) > 0:
            with torch.no_grad():
                vis_limit = 100
                vis_chunk = eval_samples[:vis_limit]
                obs_list = {k: [] for k in vis_chunk[0][0].keys()}
                gt_acts = []
                for obs, act in vis_chunk:
                    for k, v in obs.items():
                        obs_list[k].append(v)
                    gt_acts.append(act)
                for k in obs_list:
                    arr = np.asarray(obs_list[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    obs_list[k] = torch.as_tensor(arr, device=config.device).float()
                
                enc_in = {k: v for k, v in obs_list.items()}
                embedding = encoder(enc_in)
                pred_acts = action_mlp(embedding).cpu().numpy()
                gt_acts = np.array(gt_acts)

                action_dim = gt_acts.shape[1]
                fig, axes = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim), sharex=True)
                if action_dim == 1:
                    axes = [axes]
                for i in range(action_dim):
                    axes[i].plot(gt_acts[:, i], label='Ground Truth', color='black', alpha=0.6)
                    axes[i].plot(pred_acts[:, i], label='Predicted', color='red', linestyle='--', alpha=0.6)
                    axes[i].set_ylabel(f'Dim {i}')
                    if i == 0:
                        axes[i].legend()
                plt.tight_layout()
                wandb.log({"eval/action_plots": wandb.Image(fig)}, commit=False)
                plt.close(fig)

        action_mlp.train()
        return float(np.mean(losses)) if losses else None

    global_step = 0
    for epoch in range(1, config.bc_epochs + 1):
        if config.bc_shuffle:
            np.random.shuffle(train_samples)
        epoch_losses = []
        for step in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            batch_obs, batch_actions = _sample_batch(train_samples, config.bc_batch_size, config.device)
            with torch.no_grad():
                embedding = encoder(batch_obs)
            pred_actions = action_mlp(embedding)
            loss = loss_fn(pred_actions, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_mlp.parameters(), config.bc_grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
            global_step += 1
        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        eval_loss = evaluate() if (epoch % config.bc_eval_every == 0) else None
        msg = f"Epoch {epoch}/{config.bc_epochs} | train_loss={avg_train_loss:.4f}"
        if eval_loss is not None:
            msg += f" | eval_loss={eval_loss:.4f}"
        print(msg)
        wandb.log({"train/loss": avg_train_loss, "epoch": epoch, **({"eval/loss": eval_loss} if eval_loss is not None else {})})
        if global_step >= total_steps:
            print("Reached max BC steps; stopping early.")
            break
    print("BC training complete.")
    wandb.finish()
    return action_mlp

if __name__ == "__main__":
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--configs", nargs="+")
    args, remaining = base_parser.parse_known_args()
    config = _load_config(args.configs, remaining)
    encoder, action_mlp = BC_MLP(config)
    trained_mlp = BC_MLP_train(config, encoder, action_mlp)

    save_path = config.bc_save_path or (pathlib.Path(config.logdir) / "bc_action_mlp.pt")
    save_path = pathlib.Path(save_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_mlp_state_dict": trained_mlp.state_dict(),
            "config": vars(config),
            "encoder_outdim": getattr(encoder, "outdim", None),
        },
        save_path,
    )
    print(f"Saved behavior cloning MLP to {save_path}")
