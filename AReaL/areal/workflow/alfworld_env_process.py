# SPDX-License-Identifier: MIT
"""Run one disposable TextWorld process; keep imports free of AReaL, torch, and tokenizers."""

from __future__ import annotations

import argparse
import faulthandler
import os
import resource
import signal
import subprocess
import sys
import traceback
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path


class EnvironmentProcessError(RuntimeError):
    pass


class IsolatedTextWorldEnv:
    process_isolated = True

    def __init__(self, repo_root, gamefile, max_steps, *, timeout=180, _script=None):
        self.timeout = timeout
        self._closed = False
        self._connection, child = Pipe(duplex=True)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    str(_script or Path(__file__).resolve()),
                    "--connection-fd",
                    str(child.fileno()),
                    "--repo-root",
                    str(repo_root),
                ],
                pass_fds=(child.fileno(),),
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()
        try:
            self.runtime = self._call("create", gamefile, max_steps)
        except BaseException:
            self.close()
            raise

    def _call(self, operation, *args):
        if self._closed:
            raise EnvironmentProcessError("Environment process is closed")
        try:
            self._connection.send((operation, args))
            if not self._connection.poll(self.timeout):
                raise TimeoutError(
                    f"Environment {operation} timed out after {self.timeout}s"
                )
            ok, value = self._connection.recv()
        except (EOFError, OSError, TimeoutError) as exc:
            pid = self._process.pid
            # Capture the child's actual status before cleanup can replace it.
            try:
                code = self._process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                code = None
            self.close()
            raise EnvironmentProcessError(
                f"Environment {operation} failed (pid={pid}, exitcode={code}): {exc}"
            ) from exc
        if not ok:
            self.close()
            raise EnvironmentProcessError(f"Environment {operation}: {value}")
        return value

    def reset(self):
        return self._call("reset")

    def step(self, actions):
        return self._call("step", actions)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # This process owns exactly one environment. Releasing the process lets
        # the OS release native mappings; never dlclose live C++ library state.
        self._connection.close()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait(timeout=3)


def make_env(gamefile, max_steps):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )

    infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=False,
        extras=["gamefile", "expert_plan"],
    )
    env_id = textworld.gym.register_games(
        [gamefile],
        infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[
            AlfredDemangler(shuffle=False),
            AlfredInfos,
            AlfredExpert(AlfredExpertType.HANDCODED),
        ],
    )
    return textworld.gym.make(env_id)


def portable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    # TextWorld's batch interface can return NumPy scalars/arrays.
    if hasattr(value, "tolist"):
        return portable(value.tolist())
    raise TypeError(f"Unsupported environment reply type: {type(value).__name__}")


def observation_reply(value):
    # Return observations, commands, and success only; pickled facts would import TextWorld in the model worker.
    *items, infos = value
    kept = {
        key: item
        for key, item in infos.items()
        if key in ("admissible_commands", "won", "extra.gamefile", "extra.expert_plan")
    }
    return (*[portable(item) for item in items], portable(kept))


def serve(connection, factory):
    env = None
    try:
        while True:
            try:
                operation, args = connection.recv()
            except EOFError:
                return env
            try:
                if operation == "create" and env is None:
                    env = factory(*args)
                    heavy = sorted(
                        set(sys.modules) & {"torch", "areal", "transformers"}
                    )
                    if heavy:
                        raise RuntimeError(
                            f"Environment worker loaded model runtime: {heavy}"
                        )
                    value = {
                        "pid": os.getpid(),
                        "model_modules": heavy,
                        "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    }
                elif operation == "reset" and env is not None:
                    value = observation_reply(env.reset())
                elif operation == "step" and env is not None:
                    value = observation_reply(env.step(*args))
                else:
                    raise ValueError(f"Invalid environment operation: {operation}")
                connection.send((True, value))
            except Exception:
                connection.send((False, traceback.format_exc()))
                return env
    finally:
        connection.close()


def main(factory=make_env):
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-fd", type=int, required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    faulthandler.enable(all_threads=True)
    sys.path.insert(0, args.repo_root)
    retained = []
    retained.append(serve(Connection(args.connection_fd), factory))
    # Avoid interpreter/native finalizers: native state belongs only to this
    # disposable process and is reclaimed on exit, including abnormal parents.
    os._exit(0)


if __name__ == "__main__":
    main()
