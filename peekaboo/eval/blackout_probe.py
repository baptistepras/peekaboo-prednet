"""A check of step 4.1: does the position probe underrate what the state holds on the blackout frames?

Every probe of the project was fitted on the streams of the v1 mix, which has no blackouts. During a blackout, the
input is black, and the state may sit where the linear probe was never fitted. The check fits the same probe (the
settings of scripts/fit_probes.py) on a stream whose mix gives half of the sequences to blackouts, the other
conditions in their v1 proportions (probe_mix), so that it sees a few thousand blackout frames, then reads both
probes, the original and the new one, on the same states in open loop. The share of one half was fixed before the
check was run.

- region_rows(rows): the frames read, by region: the blackout frames of the blackout cells, the hidden frames of the
  occlusion cells (j = 0 to k - 1 since the onset in both), and the visible frames of the occlusion cells before the
  onset, where the new probe should not do worse.
- summarize(rows): one value per sequence and seed, and the mean over the seeds with a bootstrap interval (seeds as
  the outer level), per model, region, k, and probe.
"""

import pandas as pd

from peekaboo.config import apply_values
from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.occlusion import interval
from peekaboo.eval.stats import bootstrap
from peekaboo.eval.tables import per_sequence

BLACKOUT_SHARE = 0.5  # share of the probe stream given to blackouts
PROBES = {"probe": "probe_y", "probe fitted with blackouts": "new_probe_y"}  # the probes compared: their y column
REGIONS = ("blackout", "occlusion hidden", "occlusion visible")
VISIBLE = STATE_NAMES.index("visible")


def probe_mix(data_config: dict) -> dict:
    """The data config of the probe stream: the v1 mix scaled to 1 - BLACKOUT_SHARE, and blackouts for the rest."""
    mix = {name: share * (1.0 - BLACKOUT_SHARE) for name, share in data_config["conditions"]["train_mix"].items()}
    return apply_values(data_config, {"conditions.train_mix": {**mix, "blackout": BLACKOUT_SHARE}})


def region_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows of the three regions, with `region`: the frames j = 0 to k - 1 since the onset of the blackout and
    occlusion cells, and the visible frames before the onset of the occlusion cells. Other rows are dropped."""
    j = rows["since_onset"]
    hidden = (j >= 0) & (j < rows["k"])
    region = pd.Series(None, index=rows.index, dtype=object)
    region[(rows["condition"] == "blackout") & hidden] = "blackout"
    region[(rows["condition"] == "occlusion") & hidden] = "occlusion hidden"
    region[(rows["condition"] == "occlusion") & (j < 0) & (rows["state"] == VISIBLE)] = "occlusion visible"
    return rows.assign(region=region).dropna(subset=["region"]).reset_index(drop=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model, region, k (all and each), and probe: the error on y, one value per sequence and seed (its mean over
    the frames of the region), averaged over the seeds with a bootstrap interval; and the mean of each seed."""
    out = []
    for (model, region), own in rows.groupby(["model", "region"], sort=False):
        for k in ["all"] + sorted(own["k"].unique()):
            part = own if k == "all" else own[own["k"] == k]
            for name, column in PROBES.items():
                errors = part.assign(error=(part[column] - part["y"]).abs())
                values = per_sequence(errors, "error")
                seeds = {f"seed {seed}": float(value) for seed, value in values.mean(axis=1).items()}
                out.append({"model": model, "region": region, "k": str(k), "probe": name,
                            **interval(bootstrap(values)), **seeds})
    return pd.DataFrame(out)
