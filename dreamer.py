import argparse
import datetime
import functools
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def _split_env_paths(value):
    return [item for item in value.split(":") if item]


def _running_under_apptainer():
    return (
        "/.singularity.d/libs" in _split_env_paths(os.environ.get("LD_LIBRARY_PATH", ""))
        or "APPTAINER_NAME" in os.environ
        or "SINGULARITY_NAME" in os.environ
    )


def _prepare_osmesa_env_for_apptainer():
    if os.environ.get("MUJOCO_GL", "").lower() != "osmesa" or not _running_under_apptainer():
        return

    env = os.environ.copy()
    preload_paths = set(_split_env_paths(env.get("APPTAINER_CONTAINER_GLVND_PRELOAD", "")))
    kept_preload = [path for path in _split_env_paths(env.get("LD_PRELOAD", "")) if path not in preload_paths]
    if kept_preload:
        env["LD_PRELOAD"] = ":".join(kept_preload)
    else:
        env.pop("LD_PRELOAD", None)

    kept_library_paths = [
        path for path in _split_env_paths(env.get("LD_LIBRARY_PATH", "")) if path != "/.singularity.d/libs"
    ]
    if kept_library_paths:
        env["LD_LIBRARY_PATH"] = ":".join(kept_library_paths)
    else:
        env.pop("LD_LIBRARY_PATH", None)

    if (
        env.get("LD_PRELOAD") == os.environ.get("LD_PRELOAD")
        and env.get("LD_LIBRARY_PATH") == os.environ.get("LD_LIBRARY_PATH")
    ):
        return

    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prepare_osmesa_env_for_apptainer()

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

import wandb

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, expt_dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._expt_dataset = expt_dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training and self._dataset is not None:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(self._sample_train_batch())
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    video_pred = to_np(openl)
                    if video_pred.ndim == 4:
                        video_pred = video_pred[None]
                    self._logger.video("train_openl", video_pred)
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _sample_train_batch(self):
        if self._expt_dataset is None:
            return next(self._dataset)
        replay_batch = next(self._dataset)
        expert_batch = next(self._expt_dataset)
        first_key = next(iter(replay_batch))
        if torch.is_tensor(replay_batch[first_key]):
            merged_batch = {
                key: torch.cat([expert_batch[key], replay_batch[key]], dim=0)
                for key in replay_batch
            }
        else:
            merged_batch = {
                key: np.concatenate([expert_batch[key], replay_batch[key]], axis=0)
                for key in replay_batch
            }
        first_key = next(iter(merged_batch))
        merged_size = merged_batch[first_key].shape[0]
        expected_size = int(self._config.batch_size)
        if merged_size != expected_size:
            expert_size = expert_batch[first_key].shape[0]
            replay_size = replay_batch[first_key].shape[0]
            raise ValueError(
                "Merged train batch has incorrect size: "
                f"got {merged_size} (expert={expert_size}, replay={replay_size}), "
                f"expected {expected_size}."
            )
        return merged_batch

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(
    episodes,
    config=None,
    *,
    batch_size=None,
    batch_length=None,
    episode_sampling=None,
):
    if config is not None:
        if batch_size is None:
            batch_size = getattr(config, "batch_size", None)
        if batch_length is None:
            batch_length = getattr(config, "batch_length", None)
        if episode_sampling is None:
            episode_sampling = getattr(config, "dataset_episode_sampling", "shorter")
    if episode_sampling is None:
        episode_sampling = "shorter"
    if batch_size is None or batch_length is None:
        raise ValueError(
            "make_dataset requires either a config with batch_size and batch_length "
            "or explicit batch_size and batch_length."
        )
    batch_size = int(batch_size)
    batch_length = int(batch_length)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    if batch_length <= 0:
        raise ValueError(f"batch_length must be positive; got {batch_length}.")
    if hasattr(episodes, "sample_batch"):
        dataset = tools.ReplayStoreDataset(
            episodes,
            batch_size=batch_size,
            batch_length=batch_length,
            device=(getattr(config, "device", None) if config is not None else None),
        )
        return dataset
    generator = tools.sample_episodes(
        episodes, batch_length, sampling_mode=episode_sampling
    )
    dataset = tools.from_generator(generator, batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "robosuite":
        from envs.robosuite_env import make_Robosuite_env

        env = make_Robosuite_env(config, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = (
        pathlib.Path(config.traindir).expanduser()
        if config.traindir
        else logdir / "train_eps"
    )
    config.evaldir = (
        pathlib.Path(config.evaldir).expanduser()
        if config.evaldir
        else logdir / "eval_eps"
    )
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    if getattr(config, "env_restart_every", 0):
        config.env_restart_every //= config.action_repeat
    if config.eval_only:
        config.eval_episode_num = 1
    config.dataset_backend = str(
        getattr(config, "dataset_backend", "cpu")
    ).lower()
    config.dataset_gpu_fallback = str(
        getattr(config, "dataset_gpu_fallback", "hybrid")
    ).lower()
    config.dataset_store_image_dtype = str(
        getattr(config, "dataset_store_image_dtype", "uint8")
    )
    config.dataset_store_float_dtype = str(
        getattr(config, "dataset_store_float_dtype", "float32")
    )
    config.dataset_episode_sampling = str(
        getattr(config, "dataset_episode_sampling", "shorter")
    ).lower()

    def _episode_stats(episodes):
        if not episodes:
            return 0, 0
        transitions = 0
        for episode in episodes.values():
            try:
                transitions += max(0, len(episode["reward"]) - 1)
            except Exception:
                continue
        return len(episodes), transitions

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)

    # --- Weights & Biases init; sync existing TensorBoard logs automatically ---
    os.environ.setdefault("WANDB_LOGDIR", str(logdir))
    date_tag = datetime.datetime.now().strftime("%Y%m%d")
    logdir_name = logdir.name or "logdir"
    default_wandb_name = f"{config.task}-{logdir_name}-{date_tag}"
    wandb_name = os.getenv("WANDB_NAME", default_wandb_name)
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "dreamerv3_can"),
        entity=os.getenv("WANDB_ENTITY"),                 # optional
        name=wandb_name,
        config=vars(config),                              # capture all flags
        dir=str(logdir),                                  # keep run files in logdir
        sync_tensorboard=True,                            # mirror TB scalars/images/videos to W&B
        mode=os.getenv("WANDB_MODE", "online"),           # set WANDB_MODE=offline if needed
    )

    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)
    logger.attach_wandb(wandb, run)



    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    expert_fraction = float(np.clip(getattr(config, "expert_data_fraction", 0.0), 0.0, 1.0))
    expert_batch_size = int(round(config.batch_size * expert_fraction))
    expert_batch_size = int(np.clip(expert_batch_size, 0, int(config.batch_size)))
    replay_batch_size = int(config.batch_size) - expert_batch_size

    expt_eps = None
    if not config.eval_only and expert_batch_size > 0:
        exptdir = getattr(config, "exptdir", None)
        if not exptdir:
            raise ValueError(
                "expert_data_fraction requires expert data, but config.exptdir is not set."
            )
        expt_eps = tools.load_episodes(exptdir, limit=30000)
        if not expt_eps:
            raise ValueError(f"No expert episodes found in {exptdir}.")

    use_gpu_dataset = (
        (not config.eval_only)
        and config.dataset_backend == "gpu"
    )
    train_ep_count, train_transitions = _episode_stats(train_eps)
    eval_ep_count, eval_transitions = _episode_stats(eval_eps)
    expert_ep_count, expert_transitions = _episode_stats(expt_eps)
    print(
        "[startup] dataset backend="
        f"{config.dataset_backend}, gpu_fallback={config.dataset_gpu_fallback}, "
        f"store_image_dtype={config.dataset_store_image_dtype}, "
        f"store_float_dtype={config.dataset_store_float_dtype}, "
        f"episode_sampling={config.dataset_episode_sampling}"
    )
    print(
        "[startup] expert fraction="
        f"{expert_fraction:.3f}, replay_batch_size={replay_batch_size}, "
        f"expert_batch_size={expert_batch_size}"
    )
    print(
        f"[startup] loaded train episodes={train_ep_count}, transitions={train_transitions}"
    )
    print(
        f"[startup] loaded eval episodes={eval_ep_count}, transitions={eval_transitions}"
    )
    if expert_batch_size > 0:
        print(
            f"[startup] loaded expert episodes={expert_ep_count}, transitions={expert_transitions}"
        )
    else:
        print("[startup] expert dataset disabled by expert_data_fraction=0.0")

    def _build_replay_store(episodes, name):
        fallback_mode = config.dataset_gpu_fallback
        if fallback_mode == "hybrid":
            store = tools.HybridReplayStore(config, name=name)
        elif fallback_mode in ("none", "gpu", "strict"):
            store = tools.GpuReplayStore(config, name=name)
        else:
            raise ValueError(
                f"Unsupported dataset_gpu_fallback: {config.dataset_gpu_fallback}"
            )
        for episode_id, episode in episodes.items():
            store.add_episode(episode_id, episode)
        store.evict_to_limit(config.dataset_size)
        return store

    replay_store = None
    expert_store = None
    train_cache = train_eps
    if use_gpu_dataset:
        replay_store = _build_replay_store(train_eps, name="replay")
        replay_stats = replay_store.stats()
        print(
            "[startup] replay store "
            f"gpu_eps={replay_stats['replay_gpu_episodes']} "
            f"spill_eps={replay_stats['replay_spill_episodes']} "
            f"gpu_transitions={replay_stats['replay_gpu_transitions']} "
            f"spill_transitions={replay_stats['replay_spill_transitions']} "
            f"oom_fallbacks={replay_stats['replay_oom_fallbacks']}"
        )
        train_eps.clear()
        if expt_eps is not None:
            expert_store = _build_replay_store(expt_eps, name="expert")
            expert_stats = expert_store.stats()
            print(
                "[startup] expert store "
                f"gpu_eps={expert_stats['replay_gpu_episodes']} "
                f"spill_eps={expert_stats['replay_spill_episodes']} "
                f"gpu_transitions={expert_stats['replay_gpu_transitions']} "
                f"spill_transitions={expert_stats['replay_spill_transitions']} "
                f"oom_fallbacks={expert_stats['replay_oom_fallbacks']}"
            )
            expt_eps.clear()
        train_cache = {}
    else:
        print("[startup] using CPU episode datasets (original replay path).")

    def _on_train_episode_done(episode_id, episode_data):
        if replay_store is None:
            return {}
        replay_store.add_episode(episode_id, episode_data)
        replay_store.evict_to_limit(config.dataset_size)
        stats = replay_store.stats()
        stats["dataset_size"] = int(len(replay_store))
        stats["train_episodes"] = int(
            stats["replay_gpu_episodes"] + stats["replay_spill_episodes"]
        )
        return stats

    def _build_envs():
        make = lambda mode, id: make_env(config, mode, id)
        if config.parallel:
            train = [
                Parallel(
                    lambda cfg=config, mode="train", idx=i: make_env(cfg, mode, idx),
                    "process",
                )
                for i in range(config.envs)
            ]
            evals = [
                Parallel(
                    lambda cfg=config, mode="eval", idx=i: make_env(cfg, mode, idx),
                    "process",
                )
                for i in range(config.envs // 2)
            ]
        else:
            train = [Damy(make("train", i)) for i in range(config.envs)]
            evals = [Damy(make("eval", i)) for i in range(config.envs)]
        return train, evals

    train_envs, eval_envs = _build_envs()
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir and not config.eval_only:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_cache,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
            action_repeat=config.action_repeat,
            on_episode_done=(_on_train_episode_done if use_gpu_dataset else None),
        )
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = None
    expt_dataset = None
    if not config.eval_only:
        if expert_batch_size <= 0:
            source = replay_store if use_gpu_dataset else train_eps
            train_dataset = make_dataset(
                source,
                config=config,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
            )
            print("[startup] train dataset source: replay only")
        elif expert_batch_size >= config.batch_size:
            source = expert_store if use_gpu_dataset else expt_eps
            train_dataset = make_dataset(
                source,
                config=config,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
            )
            print("[startup] train dataset source: expert only")
        else:
            replay_source = replay_store if use_gpu_dataset else train_eps
            expert_source = expert_store if use_gpu_dataset else expt_eps
            train_dataset = make_dataset(
                replay_source,
                config=config,
                batch_size=replay_batch_size,
                batch_length=config.batch_length,
            )
            expt_dataset = make_dataset(
                expert_source,
                config=config,
                batch_size=expert_batch_size,
                batch_length=config.batch_length,
            )
            print(
                "[startup] train dataset source: mixed "
                f"(expert={expert_batch_size}, replay={replay_batch_size})"
            )
    eval_dataset = make_dataset(
        eval_eps,
        config=config,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
    )
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        expt_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    try:
        wandb.watch(agent, log="all", log_freq=500)
    except Exception:
        pass

    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    if config.eval_only:
        print("Running single evaluation rollout.")
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(
            eval_policy,
            eval_envs,
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=1,
            action_repeat=config.action_repeat,
        )
        latest_eval = tools.load_episodes(config.evaldir, limit=1)
        if latest_eval:
            _, episode = next(iter(latest_eval.items()))
            env_video = np.array(episode["image"])
            if env_video.ndim == 4:
                env_video = env_video[None]
            logger.video("eval_env", env_video)
        if config.video_pred_log:
            try:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                video_pred = to_np(video_pred)
                if video_pred.ndim == 4:
                    video_pred = video_pred[None]
                logger.video("eval_openl", video_pred)
            except StopIteration:
                pass
        logger.write(fps=True)
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass
        try:
            wandb.finish()
        except Exception:
            pass
        return

    # make sure eval will be executed once after config.steps
    next_env_restart = None
    if int(getattr(config, "env_restart_every", 0) or 0) > 0:
        # agent._step may resume from a checkpoint / existing replay, so schedule
        # restarts relative to the current step instead of assuming step=0.
        next_env_restart = int(agent._step + int(config.env_restart_every))
        print(
            "[startup] env restarts enabled: "
            f"every={int(config.env_restart_every)} next_at={int(next_env_restart)}"
        )
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
                action_repeat=config.action_repeat,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                video_pred = to_np(video_pred)
                if video_pred.ndim == 4:
                    video_pred = video_pred[None]
                logger.video("eval_openl", video_pred)
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_cache,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            action_repeat=config.action_repeat,
            on_episode_done=(_on_train_episode_done if use_gpu_dataset else None),
        )
        if next_env_restart is not None and agent._step >= next_env_restart:
            print(f"[runtime] restarting envs at step={agent._step}.")
            for env in train_envs + eval_envs:
                try:
                    env.close()
                except Exception:
                    pass
            train_envs, eval_envs = _build_envs()
            state = None
            if isinstance(train_cache, dict):
                train_cache.clear()
            next_env_restart = int(agent._step + int(config.env_restart_every))
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    try:
        wandb.finish()
    except Exception:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

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
    defaults.setdefault("eval_only", False)
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
