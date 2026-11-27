"""A trial's summary statistics during registration.

Registration hands each registered frame to a :class:`StatsWorker`. On a
GPU the worker sets up the accumulator (gain curves, frame geometry) and
takes the frames on a thread and a CUDA stream of its own, in order, while
registration matches and writes the next frames: the statistics' kernels
then run beside registration's instead of queueing before its per-frame
waits. On the CPU, where both would compete for the same cores, each step
runs in line. The steps and kernels are the same either way, so the
results are equal bit for bit.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch

log = logging.getLogger(__name__)

BACKLOG = 32
"""Frames that may wait for the statistics; registration waits beyond."""

FAILED = object()
"""Result of a step that failed (the statistics are then given up)."""

THREADED = {"cuda": True, "cpu": False}
"""Whether the statistics run on a thread of their own, per device type
(on the CPU a second thread slowed registration with statistics from 231 s
to 300 s on a 780125 trial)."""


class StatsWorker:
    """Set up and accumulate a trial's statistics, on a thread (and, on a
    GPU, a CUDA stream) of their own where :data:`THREADED` says so.

    A step that fails is logged and stops the statistics; registration goes
    on without them (:attr:`failed`, and :meth:`finish` gives
    :data:`FAILED`).

    Parameters
    ----------
    setup : callable
        Returns the accumulator
        (:class:`~.accumulate.TrialStatsAccumulator`); it runs first.
    device : torch.device or str
        The statistics' device; :data:`THREADED` decides whether they get
        a thread.
    """

    def __init__(self, setup, device="cpu"):
        """Start the worker with ``setup``."""
        self.seconds = 0.0
        self.failed = False
        self._acc = None
        self._done = None
        self._pool = None
        self._stream = None
        device = torch.device(device)
        if THREADED.get(device.type, False):
            self._slots = threading.Semaphore(BACKLOG)
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="reg-stats")
            if device.type == "cuda":
                self._stream = torch.cuda.Stream(device)
        self._submit(self._setup, None, setup)

    def _setup(self, setup) -> None:
        """Make the accumulator."""
        self._acc = setup()

    def _call(self, name: str, *args):
        """Call the accumulator's method ``name``."""
        return getattr(self._acc, name)(*args)

    def _submit(self, fn, tensors, *args) -> Future:
        """Queue a step (waits while :data:`BACKLOG` steps are queued), or
        run it now without a thread. ``tensors`` (made on the caller's
        stream) are what the step reads."""
        if self._pool is None:
            done = Future()
            done.set_result(self._step(fn, None, None, *args))
            return done
        ready = None
        if self._stream is not None:
            ready = torch.cuda.Event()
            ready.record()  # on the caller's stream: its inputs are written
        self._slots.acquire()
        try:
            return self._pool.submit(self._step, fn, ready, tensors, *args)
        except BaseException:
            self._slots.release()
            raise

    def _step(self, fn, ready, tensors, *args):
        """Run a step unless an earlier one failed; time it."""
        try:
            if self.failed:
                return FAILED
            t0 = time.perf_counter()
            try:
                if self._stream is None:
                    return fn(*args)
                with torch.cuda.stream(self._stream):
                    self._stream.wait_event(ready)
                    for t in tensors or ():  # not reused before it is read
                        t.record_stream(self._stream)
                    return fn(*args)
            except Exception:
                log.exception("summary statistics failed; continuing without")
                self.failed = True
                return FAILED
            finally:
                self.seconds += time.perf_counter() - t0
        finally:
            if self._pool is not None:
                self._slots.release()

    def add(self, k: int, regs, var_factor, motion) -> None:
        """Queue registered frame ``k`` (see
        :meth:`~.accumulate.TrialStatsAccumulator.add`). The tensors must
        not be changed afterwards."""
        if not self.failed:
            regs = list(regs)
            tensors = [t for t in regs + [var_factor] if t.is_cuda]
            self._submit(
                self._call, tensors, "add", k, regs, var_factor, motion
            )

    def finish(self) -> Future:
        """Queue the maps after the last frame; the future's result is
        :meth:`~.accumulate.TrialStatsAccumulator.finish`'s, or
        :data:`FAILED`. Later calls return the same future."""
        if self._done is None and self.failed:
            self._done = Future()
            self._done.set_result(FAILED)
        if self._done is None:
            self._done = self._submit(self._call, None, "finish")
            if self._pool is not None:
                self._pool.shutdown(wait=False)
        return self._done

    def cancel(self) -> None:
        """Give up: queued steps are dropped (a running one completes on
        its own)."""
        self.failed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
