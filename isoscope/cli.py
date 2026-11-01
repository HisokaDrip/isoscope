import argparse
import csv
import sys
import time
from collections import deque

import numpy as np

from . import bench
from .detector import StreamDetector
from .metrics import event_report
from .stream import CHANNELS, server_metrics

BARS = " ▁▂▃▄▅▆▇█"


def sparkline(values, width=100):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return ""
    chunks = np.array_split(v, min(width, v.size))
    # min per chunk, not mean, otherwise a 4-tick spike vanishes
    pts = np.array([c.min() for c in chunks])
    lo, hi = np.percentile(pts, 1), pts.max()
    idx = np.clip((pts - lo) / (hi - lo + 1e-9) * (len(BARS) - 1), 0, len(BARS) - 1)
    return "".join(BARS[int(i)] for i in idx)


def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    try:
        X = np.array([[float(c) for c in r] for r in body])
    except ValueError:
        sys.exit("csv needs a header row and numeric columns only")
    return X, header


def episodes(flags):
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def cmd_stream(args):
    if args.csv:
        X, names = load_csv(args.csv)
        events = []
    else:
        X, events = server_metrics(n=args.n, seed=args.seed)
        names = list(CHANNELS)

    det = StreamDetector(k=args.k, min_run=args.min_run, window=args.window, random_state=args.seed)
    recent = deque(maxlen=200)
    scores, flags, culprit = [], [], {}
    open_since = None
    for t, x in enumerate(X):
        s, hit = det.step(x)
        scores.append(np.nan if s is None else s)
        flags.append(hit)
        if hit and open_since is None:
            open_since = t
            # which channel looks the most off vs the last ~200 normal ticks?
            # crude, but it's the first thing you'd want on an alert
            ref = np.array(recent)
            z = (x - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-9)
            j = int(np.argmax(np.abs(z)))
            culprit[t] = f"{names[j]} z={z[j]:+.1f}"
            if args.live:
                print(f"  t={t:>5}  ALERT  {culprit[t]}")
        elif not hit:
            open_since = None
            recent.append(x)
        if args.live:
            time.sleep(args.delay)

    eps = episodes(flags)
    print(f"streamed {len(X)} ticks x {X.shape[1]} channels ({', '.join(names)})")
    print(f"score  {sparkline(scores)}")
    print(f"       (low = weird; first {det.model.window} ticks are warmup)\n")

    if not args.live:
        print("alerts (merged consecutive ticks):")
        for a, b in eps:
            tag = _event_name(a, events)
            print(f"  t={a:>5}  len={b - a:>3}  {culprit.get(a, ''):<22} {tag}")

    if events:
        caught, fa = event_report(flags, events)
        print("\ninjected events:")
        for ev, c in zip(events, caught):
            status = f"caught at t={c} (+{c - ev['start']})" if c is not None else "MISSED"
            print(f"  {ev['kind']:<24} t={ev['start']:>5}..{ev['end']:<5} {status}")
        n_fa = sum(1 for a, b in eps if not _in_event(a, events))
        print(f"\n{sum(c is not None for c in caught)}/{len(events)} events caught, "
              f"{n_fa} false-alarm episodes ({fa} ticks, {fa / len(X):.1%} of the stream)")


def _in_event(t, events, lag=10):
    return any(e["start"] <= t < e["end"] + lag for e in events)


def _event_name(t, events, lag=10):
    for e in events:
        if e["start"] <= t < e["end"] + lag:
            return f"<- {e['kind']}"
    return ""


def cmd_bench(args):
    rows = bench.run(seeds=tuple(range(args.seeds)), n_estimators=args.trees)
    print(bench.format_rows(rows))
    print(f"\nmean ROC AUC over {args.seeds} seeds, {args.trees} trees, fit+score time")


def main(argv=None):
    p = argparse.ArgumentParser(prog="isoscope", description="isolation-based anomaly detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="batch isolation forest vs sklearn on synthetic data")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--trees", type=int, default=100)
    b.set_defaults(fn=cmd_bench)

    s = sub.add_parser("stream", help="streaming detection with half-space trees")
    s.add_argument("--csv", help="stream your own numeric csv instead of the fake metrics")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--k", type=float, default=5.0, help="MADs below median to alert")
    s.add_argument("--min-run", type=int, default=2, help="low ticks in a row before alerting")
    s.add_argument("--window", type=int, default=600)
    s.add_argument("--live", action="store_true", help="print alerts as they happen")
    s.add_argument("--delay", type=float, default=0.002)
    s.set_defaults(fn=cmd_stream)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
