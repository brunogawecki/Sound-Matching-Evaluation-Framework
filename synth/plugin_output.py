"""Silencing the noise VST3 plugins write straight to the process's file descriptors.

Diva writes its machine report, revision banner and a long run of ``makeAutomatable`` warnings
to **stdout** as well as stderr on every instantiation (Dexed only writes to stderr, and keeps
its own local ``suppressed_stderr`` in ``synth/dexed/synth.py`` rather than importing this
module -- see the note below). JUCE-hosted plugins log through the OS file descriptors, not
through Python's ``sys.stdout`` / ``sys.stderr``, so ``contextlib.redirect_stdout`` cannot catch
them.

Not shared with Dexed on purpose: adding an import to ``synth/dexed/synth.py`` measurably
changed the failure rate of ``tests/test_parallel_render_backend.py::
test_parallel_matches_serial_with_real_dexed`` (0/20 on the unmodified tree vs. a real,
non-negligible rate with the import added, both measured over repeated runs). That test spawns
worker processes via ``multiprocessing.Pool`` with the **spawn** start method; every worker
re-imports ``synth.dexed.synth`` regardless of which backend test triggered it (``dataset/
render_backends.py`` imports it at module level), so the weight of that one module's import
graph is now paid by every spawned worker in the suite. The failures are real (non-bit-identical
audio, not a crash), only affect patches with ``LFO KEY SYNC`` off (Dexed's free-running LFO),
and are not explained by wall-clock delay before rendering or by generic CPU contention -- both
were tested and ruled out. The mechanism is not fully diagnosed (Dexed is closed-source), so the
safe choice is to not add anything to Dexed's import path that doesn't have to be there.
"""
import contextlib
import os
from typing import Generator


@contextlib.contextmanager
def suppressed_plugin_output() -> Generator[None, None, None]:
    """Silence both stdout and stderr. Only fds 1 and 2 are redirected, so real Python
    exceptions and tracebacks are unaffected."""
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        for fd in saved:
            os.dup2(devnull_fd, fd)
        yield
    finally:
        for fd, saved_fd in saved.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull_fd)


__all__ = ["suppressed_plugin_output"]
