# convert_robomimic_to_dreamer.py
import argparse, h5py, numpy as np, pathlib, os
from collections import OrderedDict

def concat_cams(obs_group, keys):
    imgs = [obs_group[k][()] for k in keys]  # each (T,H,W,3) uint8
    # sanity check same HWC
    H,W,C = imgs[0].shape[1:]
    for im in imgs[1:]:
        assert im.shape[1:] == (H,W,C)
    return np.concatenate(imgs, axis=-1)     # (T,H,W,3*K) uint8

def build_episode(g, img_keys, lowdim_keys, aux_only_keys=None):
    """
    g = f['data/demo_xx']
    returns dict of arrays with length T+1
    """
    T = g['actions'].shape[0]
    obs = g['obs']
    nxt = g['next_obs']

    # images
    if img_keys:
        obs_img = concat_cams(obs, img_keys)           # (T,H,W,3K)
        nxt_img = concat_cams(nxt, img_keys)           # (T,H,W,3K)
        image = np.zeros((T+1,)+obs_img.shape[1:], dtype=np.uint8)
        image[0]      = obs_img[0]
        image[1:T+1]  = nxt_img[:]                      # shift by +1
    else:
        image = None

    # low-dim keys (float32)
    low = {}
    aux_only_keys = aux_only_keys or []
    for k in list(lowdim_keys) + list(aux_only_keys):
        o = obs[k][()].astype(np.float32)              # (T,D)
        n = nxt[k][()].astype(np.float32)              # (T,D)
        arr = np.zeros((T+1, o.shape[-1]), dtype=np.float32)
        arr[0]     = o[0]
        arr[1:T+1] = n[:]
        name = f"aux_{k}" if k in aux_only_keys else k
        low[name] = arr

    # actions: dummy at 0, then actions[t]
    A = g['actions'][()].astype(np.float32)            # (T,A)
    action = np.zeros((T+1, A.shape[-1]), dtype=np.float32)
    action[1:T+1] = A

    # rewards/discount/is_first/is_terminal
    r = g['rewards'][()].astype(np.float32)            # (T,)
    d = g['dones'][()].astype(np.uint8)                # (T,)
    reward      = np.zeros((T+1,), dtype=np.float32);  reward[1:T+1] = r
    discount    = np.ones((T+1,), dtype=np.float32);   discount[1:T+1] = 1.0 - d
    is_first    = np.zeros((T+1,), dtype=np.uint8);    is_first[0] = 1
    is_terminal = np.zeros((T+1,), dtype=np.uint8);    is_terminal[1:T+1] = d

    ep = OrderedDict()
    if image is not None: ep['image'] = image
    for k,v in low.items(): ep[k] = v
    ep['action']      = action
    ep['reward']      = reward
    ep['discount']    = discount
    ep['is_first']    = is_first
    ep['is_terminal'] = is_terminal
    return ep

def save_episode_npz(outdir, name, episode):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    length = len(episode['reward'])  # T+1
    path = outdir / f"{name}-{length}.npz"
    np.savez_compressed(path, **episode)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_keys", default="agentview_image")
    ap.add_argument("--lowdim_keys", default="robot0_joint_pos,robot0_joint_vel,robot0_gripper_qpos,robot0_gripper_qvel")
    ap.add_argument("--aux_only_keys", default="robot0_joint_pos_sin,robot0_joint_pos_cos")
    args = ap.parse_args()

    img_keys = [k for k in args.img_keys.split(",") if k.strip()]
    lowdim_keys = [k for k in args.lowdim_keys.split(",") if k.strip()]
    aux_only_keys = [k for k in args.aux_only_keys.split(",") if k.strip()]

    with h5py.File(args.in_h5, "r") as f:
        for demo in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1])):
            g = f["data"][demo]
            ep = build_episode(g, img_keys, lowdim_keys, aux_only_keys)
            save_episode_npz(args.out_dir, demo, ep)
