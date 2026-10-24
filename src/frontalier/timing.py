"""
Stage banners, timings and a summary of a run.

Every heavy step announces itself before it starts -- so a long calculation is
never silent -- and reports how long it took when it is done.  At the end of a
driver, `summary()` prints the stages in the order they ran with the total.

    with stage("fragment Ti5"):
        ...
      ...
      done in 0.25 s

`timed` is the light version for calls inside an expression (no banner, one
line).  Nothing here is opt-in: the point is to always know what is running.

    FRONTALIER_QUIET=1    silence the per-call lines (banners and summary stay)
"""

import os
import time
from contextlib import contextmanager

QUIET = os.environ.get("FRONTALIER_QUIET", "").strip() not in ("", "0")
WIDTH = 74
STAGES = []          # [(title, seconds)] in the order they ran


def reset():
    """Forget the recorded stages (a test wants a clean summary)."""
    del STAGES[:]


def banner(title):
    """The `---- title ----` block that says a stage is starting."""
    print(f"\n{'-' * WIDTH}\n{title}\n{'-' * WIDTH}", flush=True)


def record(title, seconds):
    STAGES.append((title, seconds))


@contextmanager
def stage(title):
    """Banner before, elapsed time and a record after."""
    banner(title)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    record(title, elapsed)
    print(f"   done in {elapsed:.2f} s", flush=True)


@contextmanager
def timed(label):
    """A single line and a record, for calls that sit inside an expression."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    record(label, elapsed)
    if not QUIET:
        print(f"   {label:<52s}{elapsed:7.2f} s", flush=True)


def timed_call(label, function, *args, **kwargs):
    """`function(*args)` timed as one line, for calls inside an expression."""
    with timed(label):
        return function(*args, **kwargs)


def summary(title="timing summary"):
    """The stages, aggregated by label, with the total and the slowest one.

    A driver calls the same builder several times (once per origin, again
    inside the frontier model, again inside the spectra), so identical labels
    are summed and marked with their count instead of appearing three times.
    """
    if not STAGES:
        return
    total = sum(seconds for _, seconds in STAGES)
    order, summed = [], {}
    for label, seconds in STAGES:
        if label not in summed:
            order.append(label)
            summed[label] = [0.0, 0]
        summed[label][0] += seconds
        summed[label][1] += 1

    banner(title)
    for label in order:
        seconds, count = summed[label]
        share = 100.0 * seconds / total if total else 0.0
        shown = f"{label}  (x{count})" if count > 1 else label
        print(f"   {shown:<46.46s}{seconds:8.2f} s {share:5.1f} %")
    print(f"   {'-' * 46}   {'-' * 7}   {'-' * 5}")
    print(f"   {'total':<46s}{total:8.2f} s")
    slug, (seconds, _) = max(summed.items(), key=lambda item: item[1][0])
    print(f"   slowest: {slug} ({seconds:.2f} s)", flush=True)
