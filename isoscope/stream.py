"""Fake server metrics with anomalies injected at known places.

Three channels: cpu %, p95 latency (ms), requests/s. They're correlated the
way real ones are (more traffic -> more cpu -> more latency), which matters
for the 'decoupled' event: every value on its own looks fine, it's the
combination that's wrong.
"""

import numpy as np

CHANNELS = ("cpu", "latency_ms", "req_s")


def server_metrics(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    load = 1.0 + 0.35 * np.sin(2 * np.pi * t / 600) + 0.08 * rng.normal(size=n)
    req = 400 * load + rng.normal(0, 12, n)
    cpu = 18 + 0.09 * req + rng.normal(0, 2.0, n)
    lat = 40 + 0.05 * req + 0.6 * cpu + rng.normal(0, 3.0, n)
    X = np.c_[cpu, lat, req]

    events = []

    def inject(kind, start, length, fn):
        sl = slice(start, start + length)
        fn(X, sl)
        events.append({"kind": kind, "start": start, "end": start + length})

    inject("latency spike", 900, 4, lambda X, s: X.__setitem__((s, 1), X[s, 1] + 260))
    inject("cpu runaway", 1400, 30, lambda X, s: X.__setitem__((s, 0), 97 + rng.normal(0, 1, s.stop - s.start)))
    inject("traffic drop", 1900, 25, lambda X, s: X.__setitem__((s, 2), X[s, 2] * 0.05))

    def decouple(X, s):
        # cpu pegged high while traffic is low and latency normal:
        # think crypto miner / runaway cron rather than real load
        X[s, 0] = X[s, 0] + 30
        X[s, 2] = X[s, 2] * 0.6

    inject("decoupled cpu/traffic", 2400, 20, decouple)
    inject("level shift", 2700, 300, lambda X, s: X.__setitem__((s, 1), X[s, 1] + 90))
    return X, events
