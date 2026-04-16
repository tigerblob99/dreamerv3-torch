"""Offline DreamerV3 RL fine-tuning of a joint_train BC checkpoint.

Loads WM + BC actor from a joint_train checkpoint, initialises ImagBehavior
with the BC actor weights, and runs the standard DreamerV3 offline imagination
loop using the WM's own reward/cont heads as the RL objective.

python dreamer_finetune.py --configs dreamer_finetune \\
  --checkpoint logdir/joint_run/latest.pt \\
  --logdir logdir/dreamer_ft_run \\
  --offline_traindir datasets/robomimic_data_MV/can_PH_train
"""

import argparse
import copy
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import tools
import wandb
from AC_RL.RlDataset import RlDataset
from joint_train import collate_episodes, evaluate_online
from models import ImagBehavior, WorldModel
from RL_finetune import _extract_world_model_state, _init_eval_envs


class _ModePolicy(torch.nn.Module):
    """Wraps ImagBehavior.actor (networks.MLP) to match the evaluate_online interface."""
    def __init__(self, actor_mlp):
        super().__init__()
        self._actor = actor_mlp

    def forward(self, feat):
        return self._actor(feat).mode()


def _metric_scalar(value):
    tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    return float(tensor.float().mean().item())


def _load_checkpoint(wm, imag, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.load_state_dict(_extract_world_model_state(ckpt), strict=True)
    raw = ckpt.get("policy") or ckpt.get("action_mlp")
    if raw is None:
        raise KeyError("Checkpoint missing 'policy' or 'action_mlp'.")
    # joint_train saves ActionMLP under _mlp.*; ImagBehavior.actor uses Actor_* layer names.
    actor_state = {
        k[len("_mlp.") :].replace("ActionMLP_", "Actor_"): v
        for k, v in raw.items()
        if k.startswith("_mlp.")
    }
    imag.actor.load_state_dict(actor_state, strict=True)
    print(f"Loaded WM + BC actor from {ckpt_path}")


def dreamer_finetune(config):
    tools.set_seed_everywhere(int(config.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if "cuda" not in str(config.device) or not torch.cuda.is_available():
        config.precision = 32

    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, 0)

    run = None
    if wandb is not None:
        try:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "dreamer_finetune"),
                entity=os.getenv("WANDB_ENTITY"),
                name=os.getenv("WANDB_NAME", logdir.name),
                config=vars(config),
                dir=str(logdir),
                mode=os.getenv("WANDB_MODE", "online"),
            )
            logger.attach_wandb(wandb, run)
        except Exception:
            run = None

    dataset = RlDataset(config.offline_traindir, config, mode="train")
    if dataset.num_episodes == 0:
        raise RuntimeError(f"No episodes found in {config.offline_traindir}.")

    sample_ep = dataset.episode_list[0]
    obs_space, act_space = tools._define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]

    wm = WorldModel(obs_space, act_space, 0, config).to(config.device)
    imag = ImagBehavior(config, wm).to(config.device)
    _load_checkpoint(wm, imag, str(config.checkpoint), config.device)
    bc_kl_scale = float(getattr(config, "bc_kl_scale", 0.0))
    frozen_bc_actor = None
    if bc_kl_scale > 0.0:
        frozen_bc_actor = copy.deepcopy(imag.actor)
        frozen_bc_actor.requires_grad_(False)
        frozen_bc_actor.eval()
        print(f"BC-KL enabled: scale={bc_kl_scale}")
    wm.requires_grad_(False)
    imag.requires_grad_(False)

    freeze_wm = bool(getattr(config, "freeze_wm", True))
    if freeze_wm:
        wm.eval()
    else:
        wm.train()

    use_amp = bool(
        config.precision == 16
        and torch.cuda.is_available()
        and "cuda" in str(config.device)
    )
    amp_device = "cuda" if use_amp else "cpu"

    pin = bool(torch.cuda.is_available() and "cuda" in str(config.device))
    sampler = WeightedRandomSampler(
        dataset.sample_weights, num_samples=len(dataset), replacement=True
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        sampler=sampler,
        num_workers=int(getattr(config, "num_workers", 0)),
        collate_fn=collate_episodes,
        pin_memory=pin,
        drop_last=True,
    )

    def _inf(ldr):
        while True:
            yield from ldr

    batch_iter = _inf(loader)
    eval_enabled = (
        int(getattr(config, "eval_episodes", 0)) > 0
        and int(getattr(config, "num_envs", 0)) > 0
    )
    if eval_enabled and not str(getattr(config, "robosuite_task", "")).strip():
        raise ValueError("robosuite_task must be set when evaluation is enabled.")
    eval_envs = _init_eval_envs(config) if eval_enabled else None
    mode_policy = _ModePolicy(imag.actor)

    reward_fn = lambda f, s, a: wm.heads["reward"](wm.dynamics.get_feat(s)).mode()

    imag.train()
    try:
        for step in range(int(config.steps)):
            raw = next(batch_iter)
            batch = {k: v.to(config.device) for k, v in raw.items()}
            metrics = {}

            if freeze_wm:
                data = wm.preprocess(batch)
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=amp_device, enabled=use_amp, dtype=torch.float16
                    ):
                        embed = wm.encoder(data)
                        post, _ = wm.dynamics.observe(
                            embed, data["action"], data["is_first"]
                        )
            else:
                post, _, wm_metrics = wm._train(batch)
                metrics.update(
                    {
                        f"wm/{name}": _metric_scalar(value)
                        for name, value in wm_metrics.items()
                    }
                )

            imag_feat, _, _, _, imag_metrics = imag._train(post, reward_fn)
            metrics.update(imag_metrics)

            if frozen_bc_actor is not None:
                with tools.RequiresGrad(imag.actor):
                    with torch.amp.autocast(
                        device_type=amp_device, enabled=use_amp, dtype=torch.float16
                    ):
                        actor_dist = imag.actor(imag_feat.detach())
                        with torch.no_grad():
                            bc_dist = frozen_bc_actor(imag_feat.detach())
                        # Both dists are ContDist wrapping Independent(Normal, 1).
                        # .base_dist delegates via __getattr__ to Independent.base_dist
                        # which is the raw Normal — registered for kl_divergence.
                        kl = torch.distributions.kl_divergence(
                            actor_dist.base_dist, bc_dist.base_dist
                        )  # (horizon, B*T, action_dim)
                        raw_kl = kl.sum(dim=-1).mean()
                        bc_kl_loss = bc_kl_scale * raw_kl
                    imag._actor_opt(bc_kl_loss, imag.actor.parameters())
                metrics["actor_bc_kl"] = float(raw_kl.detach().float().item())
                metrics["actor_bc_kl_loss"] = float(bc_kl_loss.detach().float().item())

            if int(config.log_every) > 0 and step % int(config.log_every) == 0:
                for name, value in metrics.items():
                    logger.scalar(name, value)
                logger.step = step
                logger.write(fps=False)

            if eval_envs and int(config.eval_every) > 0 and step % int(config.eval_every) == 0:
                online_metrics = evaluate_online(wm, mode_policy, config, step, run, eval_envs)
                for name, value in online_metrics.items():
                    logger.scalar(name, value)
                logger.step = step
                logger.write(fps=False)
                imag.train()
                if freeze_wm:
                    wm.eval()
                else:
                    wm.train()

            if int(config.save_every) > 0 and step % int(config.save_every) == 0:
                torch.save(
                    {"wm": wm.state_dict(), "imag": imag.state_dict(), "step": step},
                    logdir / "latest.pt",
                )
    finally:
        if eval_envs:
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
    cfgs = yaml.safe_load(cfg_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, cfgs[name])

    complex_defaults = {}
    if args.env_config:
        for k, v in cfgs.get(args.env_config, {}).items():
            (complex_defaults if isinstance(v, (dict, list)) else defaults)[k] = v

    defaults.setdefault("checkpoint", "")
    defaults.setdefault("freeze_wm", True)
    defaults.setdefault("num_workers", 0)
    defaults.setdefault("eval_every", 5000)
    defaults.setdefault("eval_episodes", 0)
    defaults.setdefault("num_envs", 1)
    defaults.setdefault("max_env_steps", 500)
    defaults.setdefault("clip_actions", False)

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
    dreamer_finetune(config)
