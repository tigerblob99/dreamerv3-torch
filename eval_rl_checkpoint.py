"""Run N online evals of an RL_finetune checkpoint and log to W&B.

python eval_rl_checkpoint.py --ckpt logdir/<run>/rl_latest.pt --n 5
"""
import argparse
import datetime
import os
import pathlib
from types import SimpleNamespace

import numpy as np
import torch
import wandb

os.environ.setdefault("MUJOCO_GL", "osmesa")

import tools
from AC_RL.actor import Actor
from joint_train import evaluate_online
from models import WorldModel
from RL_finetune import _init_eval_envs, _load_actor_only, _load_world_model_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to rl_latest.pt")
    ap.add_argument("--n", type=int, default=5, help="Number of eval runs")
    ap.add_argument("--eval_episodes", type=int, default=None)
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--video_episodes", type=int, default=None)
    ap.add_argument("--max_env_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wm_ckpt", default=None, help="Override WM checkpoint path")
    ap.add_argument("--wandb_name", default=None)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    saved = state.get("config") or {}
    if not saved:
        raise ValueError(f"{args.ckpt} has no 'config' dict.")
    config = SimpleNamespace(**saved)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.eval_episodes is not None:
        config.eval_episodes = int(args.eval_episodes)
    if args.num_envs is not None:
        config.num_envs = int(args.num_envs)
    if args.video_episodes is not None:
        config.eval_video_episodes = int(args.video_episodes)
    if args.max_env_steps is not None:
        config.max_env_steps = int(args.max_env_steps)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.logdir = str(pathlib.Path(args.ckpt).parent / f"eval_{stamp}")
    pathlib.Path(config.logdir).mkdir(parents=True, exist_ok=True)

    tools.set_seed_everywhere(int(config.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(config.device)

    expert_dir = getattr(config, "expert_dir", "") or getattr(config, "offline_traindir", "")
    episodes = tools.load_episodes(expert_dir, limit=1)
    if not episodes:
        raise RuntimeError(f"No episodes in {expert_dir}")
    sample_ep = next(iter(episodes.values()))
    obs_space, act_space = tools._define_spaces(sample_ep, config)
    config.num_actions = act_space.shape[0]

    feat_size = (
        config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        if config.dyn_discrete
        else config.dyn_stoch + config.dyn_deter
    )
    wm = WorldModel(obs_space, act_space, 0, config).to(device)
    actor = Actor(config, feat_size, config.num_actions).to(device)

    wm_path = args.wm_ckpt or str(getattr(config, "dreamer_wm_checkpoint", "")).strip() \
        or str(getattr(config, "checkpoint", "")).strip()
    if not wm_path:
        raise FileNotFoundError("No WM checkpoint path available.")
    _load_world_model_only(wm, wm_path, device, source_label="wm_ckpt")
    _load_actor_only(actor, args.ckpt, device)
    wm.eval(); actor.eval()

    task = str(getattr(config, "robosuite_task", "task"))
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "RL_finetune_eval"),
        entity=os.getenv("WANDB_ENTITY"),
        name=args.wandb_name or f"{task}_eval_{stamp}",
        config={"ckpt": args.ckpt, "n_runs": args.n, "wm_ckpt": wm_path, **saved},
        dir=config.logdir,
    )

    eval_envs = _init_eval_envs(config)
    if eval_envs is None:
        raise RuntimeError("Could not init eval envs (missing fields in saved config?).")

    try:
        rates = []
        for i in range(args.n):
            print(f"=== Eval {i + 1}/{args.n} ===")
            m = evaluate_online(wm, actor.actionMLP, config, i, run, eval_envs)
            run.log({k: float(v) for k, v in m.items()}, step=i)
            sr = float(m.get("eval_online/success_rate", 0.0))
            ret = float(m.get("eval_online/mean_return", 0.0))
            rates.append(sr)
            print(f"  success_rate={sr:.3f}  mean_return={ret:.3f}")

        rates = np.asarray(rates, dtype=np.float64)
        std = float(rates.std(ddof=1)) if len(rates) > 1 else 0.0
        run.log({
            "summary/success_rate_mean": float(rates.mean()),
            "summary/success_rate_std": std,
            "summary/success_rate_min": float(rates.min()),
            "summary/success_rate_max": float(rates.max()),
            "summary/n_runs": int(args.n),
        }, step=args.n)
        print(f"\nsuccess_rate over {args.n} runs: mean={rates.mean():.3f}  std={std:.3f}")
    finally:
        for env in eval_envs:
            try: env.close()
            except Exception: pass
        run.finish()


if __name__ == "__main__":
    main()
