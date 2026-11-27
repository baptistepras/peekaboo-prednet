"""Tests for the check of step 4.1 that refits the position probe with blackouts: the mix of its stream, the frames
read in each region, and the summary over seeds on a fake table with known answers."""

import numpy as np
import pandas as pd
import pytest

from peekaboo.config import load_config
from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.blackout_probe import BLACKOUT_SHARE, probe_mix, region_rows, summarize
from peekaboo.paths import CONFIGS_DIR


def test_probe_mix_keeps_the_v1_proportions() -> None:
    """Half of the stream is blackouts, the rest keeps the v1 proportions, and the input config is not modified."""
    config = load_config(CONFIGS_DIR / "data" / "base.yaml")
    v1 = dict(config["conditions"]["train_mix"])
    mix = probe_mix(config)["conditions"]["train_mix"]
    assert mix["blackout"] == BLACKOUT_SHARE == 0.5 and sum(mix.values()) == pytest.approx(1.0)
    assert all(mix[name] == pytest.approx(0.5 * share) for name, share in v1.items())
    assert config["conditions"]["train_mix"] == v1


def frames(condition: str, sequence: int, k: int = 3, onset: int = 10) -> pd.DataFrame:
    """One sequence of a cell: frames 5 to 19, visible except from the onset to the reappearance."""
    t = np.arange(5, 20)
    j = t - onset
    state = np.where((j >= 0) & (j < k), STATE_NAMES.index("blackout" if condition == "blackout" else "occluded"),
                     STATE_NAMES.index("visible"))
    return pd.DataFrame({"sequence": sequence, "t": t, "condition": condition, "k": k, "since_onset": j,
                         "state": state, "y": 30.0})


def test_regions() -> None:
    """The blackout frames and the hidden frames are j = 0 to k - 1; the visible frames are those before the onset of
    the occlusion cells; the frames after the reappearance and those of other conditions are dropped."""
    rows = region_rows(pd.concat([frames("blackout", 0), frames("occlusion", 1), frames("control", 2)],
                                 ignore_index=True))
    assert rows.loc[rows["region"] == "blackout", "since_onset"].tolist() == [0, 1, 2]
    assert rows.loc[rows["region"] == "occlusion hidden", "since_onset"].tolist() == [0, 1, 2]
    assert rows.loc[rows["region"] == "occlusion visible", "since_onset"].tolist() == [-5, -4, -3, -2, -1]
    assert set(rows["sequence"]) == {0, 1}


def test_summary_over_seeds() -> None:
    """Each sequence counts once whatever its number of frames, the seeds are averaged, and each probe is read from
    its own column."""
    parts = []
    for seed, (first, second) in enumerate(((1.0, 3.0), (2.0, 4.0))):
        for sequence, (k, error) in enumerate(((2, first), (4, second))):
            rows = region_rows(frames("blackout", sequence, k=k))
            parts.append(rows.assign(probe_y=30.0 + error, new_probe_y=30.0 - error / 2, model="A", seed=seed))
    summary = summarize(pd.concat(parts, ignore_index=True)).set_index(["region", "k", "probe"])
    row = summary.loc[("blackout", "all", "probe")]
    assert row["mean"] == pytest.approx(2.5) and row["seed 0"] == pytest.approx(2.0)
    assert row["seed 1"] == pytest.approx(3.0)
    assert summary.loc[("blackout", "all", "probe fitted with blackouts"), "mean"] == pytest.approx(1.25)
    assert summary.loc[("blackout", "4", "probe"), "mean"] == pytest.approx(3.5)
