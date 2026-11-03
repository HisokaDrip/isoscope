# isoscope

Isolation-based anomaly detection written from scratch in NumPy. It has two halves:

- **Isolation Forest** (Liu et al. 2008) for batch data. Trees are stored as flat arrays, so scoring walks every sample down a tree at once instead of recursing in python. It lands within noise of scikit-learn's ROC AUC.
- **Half-Space Trees** (Tan et al. 2011) for streams. The trees are random splits of the unit cube built before any data arrives, and learning is just counting how many points land in each node. A small alerting layer sits on top (log-score, median/MAD threshold, two-low-ticks-in-a-row rule) so you get alerts instead of raw numbers.

I wanted to understand why isolation forest works so well for how dumb it sounds ("cut randomly until the point is alone"), and then whether the same trick holds up when you can't see the data in advance. It does, mostly.

## how it works

**isolation forest (`iforest.py`).** each tree is built on a random subsample of 256
rows (`max_samples`). at every node it picks a random feature that still varies and a
uniform random threshold between that feature's min and max, then recurses until a point
is alone or the depth hits `ceil(log2(256)) = 8`. outliers sit in sparse regions, so random
cuts separate them in only a few steps, while normal points need many. the depth cap is
there because past that point the tree is mostly separating normal points from each other,
which is useless here.

to score, every sample walks down a tree at the same time: nodes are flat numpy arrays
(`feature`, `threshold`, `left`, `right`, `size`), and each loop step moves all the
not-yet-at-a-leaf samples one level down. when a sample stops at a leaf that still holds
`n` points, `c(n)` (the average path length of an unsuccessful BST search) is added to
estimate the depth of the subtree that was never built. the final score is the paper's
`2 ** (-mean_path / c(256))`: near 1 is anomalous, about 0.5 or below is normal. the
`predict` threshold is the `1 - contamination` quantile of the training scores.

**half-space trees (`hstree.py`).** for streams you can't subsample in advance, so the
trees are built before seeing any data. the first window of points is used to find each
feature's range (padded by 10%) and scale inputs to roughly `[0, 1]`. each tree then
splits a randomly offset box around the unit cube in half on a random dimension, down to a
fixed depth. all trees live in `(n_trees, n_nodes)` arrays in heap order (children of `i`
are `2i+1` and `2i+2`), so one update is `depth` vectorised steps across every tree.

learning is just counting. each node has a `latest` count (current window, being filled)
and a `ref` count (last complete window). every `window` points, `latest` becomes `ref`
and a new empty `latest` starts. scoring walks a point down each tree, stops at the first
node whose `ref` count is at or below `0.1 * window` (or at the leaf), and adds up
`count * 2 ** level`. points in well-populated regions get a high score, so **low means
weird**. since only the last complete window is kept, the model forgets old data, so after
a level shift the new level stops looking anomalous once the reference window has filled
up with it.

**alerting (`detector.py`).** raw mass scores are very heavy tailed, so `StreamDetector`
takes `log(score + 1)` and keeps a rolling history of recent non-alert scores. a tick is
"low" when it falls below `median - 5 * MAD` of that history. it only alerts after 2 low
ticks in a row and after the first full window of warmup. defaults are 50 trees, depth 12,
and a 600-tick window, picked so the window covers the demo's 600-tick load cycle.

## running it

Only numpy is needed. scikit-learn is optional and is used for the comparison column and a couple of tests.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m isoscope bench           # batch iforest vs sklearn
python -m isoscope stream          # fake server metrics with 5 injected incidents
python -m isoscope stream --live   # print alerts as they fire
python -m isoscope stream --csv my_metrics.csv --window 300
pytest -q
```

`--csv` takes any file with a header row and numeric columns, one row per tick.

## the stream demo

`isoscope.stream` fakes three correlated server metrics (cpu %, p95 latency, requests/s) on a slow load cycle and injects five incidents. Most of them are easy, but the "decoupled" one isn't: cpu goes up while traffic goes down, and every channel on its own stays inside its normal range. Only the combination is off, which is exactly what a per-metric threshold misses. Output from a real run (`python -m isoscope stream`, seed 0):

```
streamed 3000 ticks x 3 channels (cpu, latency_ms, req_s)
score  ▇▇▇▆▆▆▄▅▅▆▇▇▃▇▅▆▄▄▄▃▅▇▇▇▇▇▇▇▅▅▅▅▅▃▃▇▇▇▇▇▇▇▅▃▄▄▆▇▇▇▇▇▇▇▂▁▅▆▅▆▇▇▇▇▇▅▆▅▅▅▅▆▆▇▇▁▇▇▇▄▄▄▅▆▆▇▇▃▃▃▂▃▂▁▁▂▂▃▃▃
       (low = weird; first 600 ticks are warmup)

alerts (merged consecutive ticks):
  t=  755  len=  1  latency_ms z=+2.4
  t=  901  len=  3  latency_ms z=+13.7     <- latency spike
  t= 1352  len=  1  req_s z=+1.9
  t= 1401  len= 29  cpu z=+5.4             <- cpu runaway
  t= 1625  len=  1  latency_ms z=-2.1
  t= 1675  len=  1  cpu z=-2.1
  t= 1901  len= 24  req_s z=-4.4           <- traffic drop
  t= 1951  len=  1  cpu z=+2.3
  t= 2265  len=  1  cpu z=-1.8
  t= 2401  len= 19  cpu z=+5.6             <- decoupled cpu/traffic
  t= 2564  len=  1  latency_ms z=+1.8
  t= 2701  len=299  latency_ms z=+9.2      <- level shift

injected events:
  latency spike            t=  900..904   caught at t=901 (+1)
  cpu runaway              t= 1400..1430  caught at t=1401 (+1)
  traffic drop             t= 1900..1925  caught at t=1901 (+1)
  decoupled cpu/traffic    t= 2400..2420  caught at t=2401 (+1)
  level shift              t= 2700..3000  caught at t=2701 (+1)

5/5 events caught, 7 false-alarm episodes (7 ticks, 0.2% of the stream)
```

The sparkline is the log mass score, where low means weird. The `z=` column is a quick guess at which channel is most off compared to the last ~200 normal ticks. It's crude, but it's the first thing you'd want to see on an alert. The level shift stays flagged for its whole 300 ticks because the reference window is 600 ticks, so the model hasn't had time to accept it as the new normal. Feed it longer and it does (there's a test for that).

## batch benchmark

```
dataset    shape            auc  sklearn     time  sklearn
blobs      2000x4        0.9992   0.9992    0.15s    0.08s
donut      2000x2        0.9350   0.9377    0.22s    0.08s
high_dim   3000x40       0.9547   0.9531    0.20s    0.08s

mean ROC AUC over 3 seeds, 100 trees, fit+score time
```

`donut` is the interesting one: the outliers sit in the hole in the middle of a ring, so "far from the mean" is exactly the wrong idea of anomalous. sklearn is ~2x faster because its trees are in Cython. Mine are numpy all the way down, which I'm fine with.

## things I learned the hard way

- My first HS-tree window was 250 ticks, but the load cycle is 600. Half the sine wave looked anomalous because the reference profile had never seen it. The window has to cover at least one full cycle of whatever seasonality you have.
- Mass scores are hugely heavy tailed (a few hundred for a weird point, ~10^5 for a boring one). Thresholding them raw was hopeless, and log + median/MAD fixed it.
- Requiring two consecutive low ticks cut false alarms from 73 ticks to 7 and only costs one tick of delay. That's a great trade.

## what I'd do next

- Proper per-feature attribution (path-based, like DIFFI) instead of the z-score guess
- Extended isolation forest with random hyperplane cuts. Axis-aligned cuts leave artifacts on the donut
- A tiny `psutil` collector so `stream` can watch the live machine directly

MIT licensed.
