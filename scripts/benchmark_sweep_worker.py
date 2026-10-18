"""Peak-RSS benchmark for one sweep worker on real SPY data.

Measures the per-worker resident memory of _sweep_ticker_seeded at decision-grade
n_boot so main()'s --workers can be sized from free RAM (sweep_cache.
effective_workers' peak_worker_gb). Each spawned worker has this footprint, so
the reported number is the total process RSS. Run from the repo root:

    python scripts/benchmark_sweep_worker.py

Optional args: n_boot (default 2000) n_configs (default 80). Only a subset of
the 1080-point grid is run because peak RSS is reached within the first configs
(the bootstrap index + per-config transient arrays dominate; the row list is
tiny), so a subset gives the real peak in ~2-4 min instead of ~40 min.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import strategy_sweep as ss
from sweep_cache import SweepSpec, _strict_load_frames


def _sampler(stop, out):
    try:
        import psutil
        proc = psutil.Process()
        while not stop.is_set():
            out[0] = max(out[0], proc.memory_info().rss)
            time.sleep(0.02)
    except Exception:
        pass


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_configs = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    out = [0]
    stop = threading.Event()
    t = threading.Thread(target=_sampler, args=(stop, out))
    t.start()

    spec = SweepSpec("SPY", 63, 0, n_boot, ss.TRAIN_END, 15, 63, 1000,
                     "data/raw", "data/backtest")
    frames = _strict_load_frames(spec)          # real SPY causal frames (strict)
    full_grid = ss.grid_points()
    ss.grid_points = lambda: full_grid[:n_configs]     # subset -> hit peak fast

    t0 = time.perf_counter()
    ss._sweep_ticker_seeded("SPY", frames, n_boot=n_boot, block_len=63, seed=0)
    dt = time.perf_counter() - t0
    stop.set()
    t.join()

    peak_gb = out[0] / (1024 ** 3)
    train_rows = len(frames[63][frames[63]["Date"].astype(str) <= ss.TRAIN_END])
    print(f"peak_rss_gb={peak_gb:.3f}  n_boot={n_boot}  n_configs={n_configs}  "
          f"SPY_train_rows~={train_rows}  elapsed={dt:.0f}s")


if __name__ == "__main__":
    main()
