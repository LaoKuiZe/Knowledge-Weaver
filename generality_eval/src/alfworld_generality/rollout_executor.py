"""Schedule I/O threads across bounded processes; lock TextWorld state but not actor HTTP calls."""

from __future__ import annotations

import faulthandler
import os
import pickle
import queue
import threading
import time
import traceback
from concurrent.futures import Executor, Future
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context


def _worker(incoming, outgoing, threads, initializer, initargs):
    faulthandler.enable(all_threads=True)
    if initializer is not None:
        initializer(*initargs)

    def consume():
        while True:
            payload = incoming.get()
            if payload is None:
                return
            job_id, fn, args, kwargs = pickle.loads(payload)
            try:
                value = fn(*args, **kwargs)
                result = pickle.dumps((job_id, True, value))
            except BaseException:
                # Exception objects can themselves be unpicklable. Always return
                # a serializable error; never strand a future in the parent.
                result = pickle.dumps((job_id, False, traceback.format_exc()))
            outgoing.put(result)

    workers = [threading.Thread(target=consume) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    outgoing.put(pickle.dumps((None, True, os.getpid())))


class ProcessThreadExecutor(Executor):
    """Use a shared work queue; a dead child fails pending futures and terminates siblings."""

    def __init__(
        self, *, processes, threads_per_process, initializer=None, initargs=()
    ):
        if processes <= 0 or threads_per_process <= 0:
            raise ValueError("Process and thread counts must be positive")
        context = get_context("spawn")
        self._incoming = context.Queue()
        self._outgoing = context.Queue()
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 0
        self._shutdown = False
        self._broken = None
        self._capacity = processes * threads_per_process
        self._processes = []
        try:
            for _ in range(processes):
                process = context.Process(
                    target=_worker,
                    args=(
                        self._incoming,
                        self._outgoing,
                        threads_per_process,
                        initializer,
                        initargs,
                    ),
                )
                process.start()
                self._processes.append(process)
        except BaseException:
            self._terminate()
            raise
        self._collector = threading.Thread(target=self._collect, daemon=True)
        self._collector.start()

    def submit(self, fn, /, *args, **kwargs):
        with self._lock:
            if self._broken is not None:
                raise self._broken
            if self._shutdown:
                raise RuntimeError("Cannot submit after shutdown")
            job_id = self._next_id
            # Serialize synchronously: multiprocessing.Queue's background
            # pickler otherwise drops invalid payloads without failing futures.
            payload = pickle.dumps((job_id, fn, args, kwargs))
            self._next_id += 1
            future = Future()
            future.set_running_or_notify_cancel()
            self._pending[job_id] = future
            self._incoming.put(payload)
        return future

    def _terminate(self):
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 5
        for process in self._processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.kill()
        for process in self._processes:
            process.join()
        for channel in (self._incoming, self._outgoing):
            channel.cancel_join_thread()
            channel.close()

    def _collect(self):
        clean_exits = set()
        while True:
            with self._lock:
                # Drain shutdown acknowledgements so Queue feeders can flush without deadlocking workers.
                if (
                    self._shutdown
                    and not self._pending
                    and len(clean_exits) == len(self._processes)
                ):
                    return
            try:
                job_id, ok, value = pickle.loads(self._outgoing.get(timeout=0.2))
            except queue.Empty:
                dead = [p for p in self._processes if p.exitcode is not None]
                with self._lock:
                    # Clean exits are only expected after shutdown sentinels.
                    bad = any(p.pid not in clean_exits for p in dead)
                    if not bad:
                        continue
                    self._broken = BrokenProcessPool(
                        f"Rollout worker exited (pid, exitcode): {[(p.pid, p.exitcode) for p in dead]}"
                    )
                    pending = list(self._pending.values())
                    self._pending.clear()
                for future in pending:
                    future.set_exception(self._broken)
                self._terminate()
                return
            if job_id is None:
                clean_exits.add(value)
                continue
            with self._lock:
                future = self._pending.pop(job_id)
            if ok:
                future.set_result(value)
            else:
                future.set_exception(RuntimeError(value))

    def shutdown(self, wait=True, *, cancel_futures=False):
        # Submitted jobs are marked running; cancellation never silently loses
        # an in-flight episode. The evaluator has a bounded submission queue.
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if self._broken is None:
                    for _ in range(self._capacity):
                        self._incoming.put(None)
        if wait:
            self._collector.join()
            for process in self._processes:
                process.join()
            if self._broken is None:
                for channel in (self._incoming, self._outgoing):
                    channel.close()
                    channel.join_thread()


def initialize_rollout_worker(tokenizer_path):
    """Load one tokenizer per process before any rollout threads start."""
    faulthandler.enable(all_threads=True)
    import torch

    from areal.workflow.alfworld_skill import _rollout_worker_tokenizer

    torch.set_num_threads(1)
    _rollout_worker_tokenizer(tokenizer_path)
