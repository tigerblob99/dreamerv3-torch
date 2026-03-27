"""Smoke-test for the Apptainer container environment."""

import importlib
import os
import subprocess
import sys
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = {"passed": 0, "failed": 0}


def check(name, fn):
    try:
        detail = fn()
        print(f"  [{PASS}] {name}" + (f"  ({detail})" if detail else ""))
        results["passed"] += 1
    except Exception as e:
        print(f"  [{FAIL}] {name}  -- {e}")
        traceback.print_exc()
        results["failed"] += 1


# ── 1. Python version ──────────────────────────────────────────────
print("\n== Python ==")
check("Python >= 3.11", lambda: (
    None if sys.version_info >= (3, 11)
    else (_ for _ in ()).throw(RuntimeError(f"got {sys.version}"))
) or f"{sys.version}")


# ── 2. Module imports ──────────────────────────────────────────────
REQUIRED_MODULES = [
    # Core scientific
    "numpy", "scipy", "h5py", "numba", "PIL",
    # PyTorch
    "torch", "torchvision",
    # RL / simulation
    "gym", "mujoco", "robosuite", "robomimic",
    # Video / image
    "cv2", "moviepy", "imageio",
    # Logging / visualization
    "matplotlib", "tensorboard", "wandb",
    # Config / serialization
    "einops", "ruamel.yaml", "cloudpickle",
    # Misc used in codebase
    "tqdm",
]

print("\n== Module imports ==")
for mod in REQUIRED_MODULES:
    check(f"import {mod}", lambda m=mod: importlib.import_module(m) and None)


# ── 3. PyTorch functionality ──────────────────────────────────────
print("\n== PyTorch ==")
import torch

check("torch version", lambda: torch.__version__)
check("CUDA available", lambda: (
    None if torch.cuda.is_available()
    else (_ for _ in ()).throw(RuntimeError("CUDA not available"))
) or f"{torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}")

check("cuDNN enabled", lambda: (
    None if torch.backends.cudnn.is_available()
    else (_ for _ in ()).throw(RuntimeError("cuDNN not available"))
) or f"v{torch.backends.cudnn.version()}")


def _test_tensor_ops():
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    c = a @ b
    assert c.shape == (64, 64)
    assert c.device.type == "cuda"
    return f"matmul ok, dtype={c.dtype}"


check("CUDA tensor ops", _test_tensor_ops)


def _test_autograd():
    x = torch.randn(8, 4, device="cuda", requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    return "backward pass ok"


check("autograd on CUDA", _test_autograd)


def _test_nn():
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
    ).cuda()
    x = torch.randn(4, 32, device="cuda")
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert out.shape == (4, 16)
    return "forward+backward ok"


check("nn.Module on CUDA", _test_nn)


def _test_amp():
    with torch.amp.autocast("cuda"):
        x = torch.randn(16, 16, device="cuda")
        y = torch.nn.functional.linear(x, torch.randn(16, 16, device="cuda"))
        assert y.dtype == torch.float16 or y.dtype == torch.bfloat16
    return f"autocast dtype={y.dtype}"


check("AMP autocast", _test_amp)


def _test_gru():
    gru = torch.nn.GRUCell(32, 64).cuda()
    x = torch.randn(4, 32, device="cuda")
    h = torch.zeros(4, 64, device="cuda")
    h_new = gru(x, h)
    assert h_new.shape == (4, 64)
    return "GRUCell ok (used by RSSM)"


check("GRUCell (RSSM)", _test_gru)


# ── 4. Environment variable ───────────────────────────────────────
print("\n== Environment ==")
check("MUJOCO_GL set", lambda: (
    None if os.environ.get("MUJOCO_GL")
    else (_ for _ in ()).throw(RuntimeError("MUJOCO_GL not set"))
) or os.environ["MUJOCO_GL"])


# ── 5. MuJoCo rendering ───────────────────────────────────────────
print("\n== MuJoCo ==")


def _test_mujoco():
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light/><geom type='sphere' size='0.1'/></worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    return f"mujoco {mujoco.__version__}, sim step ok"


check("MuJoCo simulation", _test_mujoco)


def _test_mujoco_render_backend(backend):
    env = dict(os.environ)
    env["MUJOCO_GL"] = backend
    code = """
import os
import mujoco

xml = "<mujoco><worldbody><light/><geom type='sphere' size='0.1'/></worldbody></mujoco>"
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
renderer.update_scene(data)
frame = renderer.render()
print(f"{os.environ['MUJOCO_GL']}:{mujoco.__version__}:{frame.shape}")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(detail or f"{backend} render subprocess failed")
    return proc.stdout.strip()


check("MuJoCo offscreen render (egl)", lambda: _test_mujoco_render_backend("egl"))
check("MuJoCo offscreen render (osmesa)", lambda: _test_mujoco_render_backend("osmesa"))


# ── Summary ────────────────────────────────────────────────────────
total = results["passed"] + results["failed"]
print(f"\n{'='*40}")
print(f"Results: {results['passed']}/{total} passed, {results['failed']} failed")
print(f"{'='*40}\n")
sys.exit(1 if results["failed"] > 0 else 0)
