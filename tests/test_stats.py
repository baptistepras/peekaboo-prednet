"""Tests for the statistics of Phase 3: bootstrap intervals of known distributions, paired differences, the seed
level, and the AUROC."""

import numpy as np
import pandas as pd

from peekaboo.eval.stats import auroc, bootstrap, bootstrap_auroc, mean_se, paired_difference


def test_bootstrap_of_a_known_distribution() -> None:
    """For 2000 draws of N(0, 1), the interval holds 0, its half width is about 1.96 / sqrt(n), and the standard
    error is about 1 / sqrt(n); NaN values are left out."""
    values = np.random.default_rng(1).normal(size=2000)
    stats = bootstrap(np.r_[values, np.nan])
    assert stats["n"] == 2000 and stats["low"] < 0 < stats["high"]
    assert abs((stats["high"] - stats["low"]) / 2 - 1.96 / np.sqrt(2000)) < 0.2 * 1.96 / np.sqrt(2000)
    assert abs(stats["se"] - 1 / np.sqrt(2000)) < 0.15 / np.sqrt(2000)
    mean, se, n = mean_se(values)
    assert np.isclose(mean, values.mean()) and np.isclose(se, values.std(ddof=1) / np.sqrt(2000)) and n == 2000


def test_paired_differences() -> None:
    """The difference of identical inputs is exactly 0; the difference is taken on the common sequences only, and a
    Series (a baseline) is subtracted from every seed."""
    a = pd.Series(np.random.default_rng(2).normal(size=50), index=np.arange(50))
    same = paired_difference(a, a)
    assert same["mean"] == 0 and same["low"] == 0 and same["high"] == 0
    shifted = paired_difference(a + 1.0, a.iloc[10:])
    assert np.isclose(shifted["mean"], 1.0) and shifted["n"] == 40
    seeds = pd.DataFrame([a + 1.0, a + 3.0])
    assert np.isclose(paired_difference(seeds, a)["mean"], 2.0)


def test_seeds_are_the_outer_level() -> None:
    """With several seeds, the standard error is that of the seed means, whatever the number of sequences."""
    rng = np.random.default_rng(3)
    seeds = pd.DataFrame([rng.normal(size=500) + shift for shift in (0.0, 1.0, 2.0)])
    mean, se, n = mean_se(seeds)
    assert n == 500 and np.isclose(mean, seeds.to_numpy().mean())
    assert np.isclose(se, seeds.mean(axis=1).std(ddof=1) / np.sqrt(3))
    stats = bootstrap(seeds, resamples=500)
    assert stats["high"] - stats["low"] > 1.0  # driven by the spread between seeds, not by the 500 sequences


def test_auroc() -> None:
    """Perfect separation gives 1, the reverse 0, identical scores 0.5, and random scores the pairwise count."""
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert auroc(np.ones(5), np.ones(5)) == 0.5
    rng = np.random.default_rng(4)
    positive, negative = rng.normal(0.5, 1, 60), rng.normal(0, 1, 60)
    pairs = (positive[:, None] > negative[None]).mean() + 0.5 * (positive[:, None] == negative[None]).mean()
    assert np.isclose(auroc(positive, negative), pairs)
    stats = bootstrap_auroc(positive, negative, resamples=300)
    assert stats["low"] <= stats["auroc"] <= stats["high"] and stats["n"] == 60
