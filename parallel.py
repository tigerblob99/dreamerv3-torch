"""Lightweight helpers to run environments in separate processes.

`Parallel(ctor, "process")` creates an env in a child process and forwards
attribute access and method calls over a pipe, so the training loop can call
`env.step(...)` / `env.reset()` as usual while the heavy simulator runs
elsewhere. `Damy` is a no-op wrapper used when `parallel=False` so the rest of
the code can use the same interface without actually spawning processes.
"""

import atexit
import os
import sys
import time
import traceback
import enum
from functools import partial as bind
import multiprocessing
import cloudpickle


def _split_env_paths(value):
    return [item for item in value.split(":") if item]


def _running_under_apptainer():
    return (
        "/.singularity.d/libs" in _split_env_paths(os.environ.get("LD_LIBRARY_PATH", ""))
        or "APPTAINER_NAME" in os.environ
        or "SINGULARITY_NAME" in os.environ
    )


def _prepare_osmesa_worker_env():
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

    if os.environ.get("DREAMER_OSMESA_WORKER_READY") == "1":
        os.environ.clear()
        os.environ.update(env)
        return

    env["DREAMER_OSMESA_WORKER_READY"] = "1"
    argv = list(getattr(sys, "orig_argv", []) or [sys.executable, *sys.argv])
    os.execvpe(argv[0], argv, env)


class Parallel:
    def __init__(self, ctor, strategy):
        self.worker = Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            else:
                return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        if state is None:
            state = ctor() if callable(ctor) else ctor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(getattr(state, name))
        elif message == PMessage.CALL:
            result = getattr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
        return state, result


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    initializers = []

    def __init__(self, fn, strategy="thread", state=False):
        if not state:
            fn = lambda s, *args, fn=fn, **kwargs: (s, fn(*args, **kwargs))
        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    def __call__(self, *args, **kwargs):
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise

    def wait(self):
        return self.impl.wait()

    def close(self):
        self.impl.close()


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):

        self._context = multiprocessing.get_context("spawn")
        self._pipe, pipe = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(
            target=self._loop, args=(pipe, fn, initializers), daemon=daemon
        )
        self._process.start()
        self._nextid = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self):
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, IOError):
            pass  # The connection was already closed.
        try:
            self._process.join(0.1)
            if self._process.exitcode is None:
                try:
                    os.kill(self._process.pid, 9)
                    time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass

    def _submit(self, message, payload=None):
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid):
        while callid not in self._results:
            try:
                message, callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                raise Exception(payload)
            assert message == Message.RESULT, message
            self._results[callid] = payload
        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function, initializers):
        try:
            _prepare_osmesa_worker_env()
            callid = None
            state = None

            initializers = cloudpickle.loads(initializers)
            function = cloudpickle.loads(function)
            [fn() for fn in initializers]
            while True:
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                message, callid, payload = pipe.recv()
                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))
                elif message == Message.STOP:
                    return
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            return
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
            return
        finally:
            try:
                pipe.close()
            except Exception:
                pass


class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive, callid):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result


class Damy:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()
