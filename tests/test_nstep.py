"""n-step return accumulation (Sutton & Barto Ch. 7) sanity checks."""
import sys
from collections import deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from train_dqn import emit_nstep

G = 0.9  # gamma


def _run(rewards, n, terminal_at_end):
    """Replay a single episode through emit_nstep; return stored transitions as
    (action, return, discount, done) where action encodes the source step index."""
    q = deque()
    out = []
    last = len(rewards) - 1
    for t, r in enumerate(rewards):
        q.append((None, t, r))  # action = step index, for identification
        done = terminal_at_end and t == last
        nxt = f"s{t + 1}"
        if done:
            for _, a, ret, _, _, d, disc in emit_nstep(q, nxt, None, True, G, flush=True):
                out.append((a, ret, disc, d))
        elif len(q) == n:
            for _, a, ret, _, _, d, disc in emit_nstep(q, nxt, None, False, G, flush=False):
                out.append((a, ret, disc, d))
    return out


def test_nstep_full_windows():
    # 4 rewards, n=3, episode truncated (not terminal -> bootstrap on all).
    out = _run([1.0, 2.0, 3.0, 4.0], n=3, terminal_at_end=False)
    by_step = {a: (ret, disc, d) for a, ret, disc, d in out}
    # step 0: r0 + g r1 + g^2 r2, bootstrap g^3
    ret, disc, d = by_step[0]
    assert abs(ret - (1 + G * 2 + G**2 * 3)) < 1e-6
    assert abs(disc - G**3) < 1e-6 and d is False
    # step 1: r1 + g r2 + g^2 r3
    ret, disc, _ = by_step[1]
    assert abs(ret - (2 + G * 3 + G**2 * 4)) < 1e-6
    assert abs(disc - G**3) < 1e-6


def test_terminal_flush_truncated_returns():
    # 4 rewards, n=3, terminal at end -> tail windows shrink, done=True, no bootstrap.
    out = _run([1.0, 2.0, 3.0, 4.0], n=3, terminal_at_end=True)
    by_step = {a: (ret, disc, d) for a, ret, disc, d in out}
    # step 1 flushed at terminal: r1 + g r2 + g^2 r3, done True
    ret, disc, d = by_step[1]
    assert abs(ret - (2 + G * 3 + G**2 * 4)) < 1e-6 and d is True
    # step 3 (last): just r3, done True
    ret, disc, d = by_step[3]
    assert abs(ret - 4.0) < 1e-6 and d is True


def test_n1_reduces_to_one_step():
    out = _run([1.0, 2.0, 3.0], n=1, terminal_at_end=False)
    for a, ret, disc, d in out:
        assert abs(ret - [1.0, 2.0, 3.0][a]) < 1e-6  # return == single reward
        assert abs(disc - G) < 1e-6                   # bootstrap discount == gamma
