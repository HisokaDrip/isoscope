import numpy as np

from isoscope.cli import episodes, main, sparkline
from isoscope.detector import StreamDetector
from isoscope.metrics import event_report
from isoscope.stream import server_metrics


def test_catches_every_injected_event_quickly():
    X, events = server_metrics(seed=7)
    det = StreamDetector(random_state=0)
    flags = np.array([det.step(x)[1] for x in X])
    caught, false_alarm_ticks = event_report(flags, events)
    assert all(c is not None for c in caught)
    assert all(c - e["start"] <= 3 for c, e in zip(caught, events))
    assert false_alarm_ticks < 0.01 * len(X)


def test_episodes():
    assert episodes([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert episodes([]) == []


def test_sparkline_keeps_short_dips():
    v = np.ones(1000)
    v[500:503] = 0.0
    line = sparkline(v, width=50)
    assert len(line) == 50
    assert line.count("▁") == 1


def test_cli_bench_and_stream(capsys, tmp_path):
    main(["bench", "--seeds", "1", "--trees", "20"])
    assert "donut" in capsys.readouterr().out

    rng = np.random.default_rng(0)
    data = rng.normal(size=(400, 2))
    data[350] = [9, 9]
    data[351] = [9, 9]
    path = tmp_path / "m.csv"
    np.savetxt(path, data, delimiter=",", header="a,b", comments="")
    main(["stream", "--csv", str(path), "--window", "150"])
    out = capsys.readouterr().out
    assert "t=  351" in out
