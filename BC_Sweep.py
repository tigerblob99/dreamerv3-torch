import argparse
import os
import sys
import pathlib
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import wandb
import gym

# Project imports (expecting these modules to be available in repo)
import tools
import networks
from offline_train import _infer_spaces
from bc_mlp.BC_MLP_train import (
    _load_config,
    BC_MLP,
    _build_encoder,
    _extract_encoder_state,
    build_action_mlp,
    _flatten_episodes,
    _sample_batch,
)


def BC_MLP_train(config, encoder, action_mlp):
    """Trains both the encoder and the action MLP using behavior cloning on offline episodes."""
    print("Preparing BC dataset...")
    train_eps = tools.load_episodes(config.offline_traindir, limit=config.dataset_size)
    eval_dir = getattr(config, "offline_evaldir", None)
    eval_eps = tools.load_episodes(eval_dir, limit=config.dataset_size) if eval_dir else {}
    if not train_eps:
        raise RuntimeError("No episodes for BC training.")
    
    # Only split training data for eval if no separate eval directory was provided
    if not eval_eps and config.bc_eval_split > 0:
        episode_items = list(train_eps.items())
        eval_cut = int(len(episode_items) * config.bc_eval_split)
        eval_eps = dict(episode_items[:eval_cut]) if eval_cut > 0 else {}
        train_eps = dict(episode_items[eval_cut:])
    print(f"BC train episodes: {len(train_eps)}, eval episodes: {len(eval_eps)}")

    train_samples = _flatten_episodes(train_eps)
    eval_samples = _flatten_episodes(eval_eps) if eval_eps else []
    # Keep eval episodes as a list for visualization of complete episodes
    eval_episodes_list = list(eval_eps.values()) if eval_eps else []
    print(f"Flattened train samples: {len(train_samples)}, eval samples: {len(eval_samples)}")

    encoder.requires_grad_(True)
    action_mlp.train()

    # Optimizer
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(action_mlp.parameters()), lr=config.bc_lr, weight_decay=config.bc_weight_decay)
    loss_fn = torch.nn.MSELoss()

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
    wandb.watch([encoder, action_mlp], log="all")
    def evaluate():
        """Evaluates the action MLP on evaluation samples and logs visualizations."""
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
                    # Normalize images to [0, 1] as done in WorldModel.preprocess()
                    if k == "image":
                        obs_list[k] = obs_list[k] / 255.0
                acts = torch.as_tensor(np.asarray(act_list), device=config.device).float()
                with torch.no_grad():
                    enc_in = {k: v for k,v in obs_list.items()}
                    embedding = encoder(enc_in)  # (batch, embed)
                    pred = action_mlp(embedding)
                    loss = loss_fn(pred, acts)
                losses.append(loss.item())

        # Visualization: sample one complete random episode
        if eval_episodes_list:
            with torch.no_grad():
                # Randomly select one complete episode
                ep_idx = np.random.randint(0, len(eval_episodes_list))
                ep = eval_episodes_list[ep_idx]
                ep_length = ep['action'].shape[0]
                
                # Build obs dict for the entire episode
                obs_list = {k: [] for k in ep.keys() if k not in ('action',) and not k.startswith('log_')}
                gt_acts = []
                for t in range(ep_length):
                    for k in obs_list.keys():
                        obs_list[k].append(ep[k][t])
                    gt_acts.append(ep['action'][t])
                
                for k in obs_list:
                    arr = np.asarray(obs_list[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    obs_list[k] = torch.as_tensor(arr, device=config.device).float()
                    # Normalize images to [0, 1] as done in WorldModel.preprocess()
                    if k == "image":
                        obs_list[k] = obs_list[k] / 255.0
                
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
                plt.suptitle(f'Episode {ep_idx} ({ep_length} steps)')
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
            embedding = encoder(batch_obs)
            pred_actions = action_mlp(embedding)
            loss = loss_fn(pred_actions, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(action_mlp.parameters()), config.bc_grad_clip)
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
