"""Statistics of the Phase 3 analyses (step 3.2), over test sequences and, when there are several, over seeds.

A per frame table has many correlated frames per sequence, so every statistic first reduces it to one value per
sequence (and per seed), then works over sequences: the unit of replication is the sequence, and the seed is the
outer level when a model has several. Comparisons between models or methods are paired: they are made on the same
sequences, which removes the variation between sequences from the difference.

- mean_se: mean and standard error, for curves with many points;
- bootstrap: mean with a percentile bootstrap interval (resampling seeds, then sequences);
- paired_difference: the same for the difference of two readouts on the same sequences;
- auroc and bootstrap_auroc: how well a score separates two kinds of sequences (step 3.4).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

LEVEL = 0.95        # coverage of every interval
RESAMPLES = 2000    # bootstrap resamples
SEED = 0            # bootstrap random seed, so that every table is reproducible
CHUNK = 250         # resamples drawn at once


@lru_cache(maxsize=2)
def resample_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """(resamples, n) indices of the resampled sequences; the same for every statistic on as many sequences, so the
    tables of an analysis share their resamples. Kept for the next call, which is often on the same sequences."""
    indices = np.random.default_rng(seed).integers(0, n, size=(resamples, n)).astype(np.int32)
    indices.flags.writeable = False
    return indices


def as_matrix(values: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Values as a (seeds, sequences) float matrix: a vector is one seed; a DataFrame has one row per seed and one
    column per sequence."""
    matrix = np.asarray(values, dtype=np.float64)
    return matrix[None] if matrix.ndim == 1 else matrix


def mean_se(values: pd.Series | pd.DataFrame | np.ndarray) -> tuple[float, float, int]:
    """Mean, standard error, and number of sequences. With one seed the standard error is over sequences; with
    several, it is over the seed means (the outer level). NaN values are left out."""
    matrix = as_matrix(values)
    seeds = np.nanmean(matrix, axis=1)
    n = int(np.isfinite(matrix).any(axis=0).sum())
    if matrix.shape[0] > 1:
        return float(seeds.mean()), float(seeds.std(ddof=1) / np.sqrt(len(seeds))), n
    finite = matrix[0][np.isfinite(matrix[0])]
    se = finite.std(ddof=1) / np.sqrt(len(finite)) if len(finite) > 1 else np.nan
    return (float(finite.mean()) if len(finite) else np.nan), float(se), n


def bootstrap(values: pd.Series | pd.DataFrame | np.ndarray, level: float = LEVEL, resamples: int = RESAMPLES,
              seed: int = SEED) -> dict[str, float]:
    """Mean of the values with its standard error and a percentile bootstrap interval. The sequences are resampled
    (the same ones for every seed, since the seeds are evaluated on the same sequences), and the seeds as well when
    there are several. NaN values are left out of each mean."""
    matrix = as_matrix(values)
    keep = np.isfinite(matrix).any(axis=0)
    matrix = matrix[:, keep]
    n_seeds, n = matrix.shape
    if n == 0:
        return {"mean": np.nan, "se": np.nan, "low": np.nan, "high": np.nan, "n": 0}
    columns = resample_indices(n, resamples, seed)
    rows = np.random.default_rng(seed + 1).integers(0, n_seeds, size=(resamples, n_seeds))
    finite = np.isfinite(matrix)
    filled = np.where(finite, matrix, 0.0)
    means = np.empty(resamples)
    if n_seeds == 1:  # by blocks of resamples, which keeps the memory at CHUNK x sequences
        for start in range(0, resamples, CHUNK):
            pick = columns[start:start + CHUNK]
            means[start:start + CHUNK] = filled[0][pick].sum(axis=1) / np.maximum(finite[0][pick].sum(axis=1), 1)
    else:
        for i in range(resamples):
            sums = filled[rows[i]][:, columns[i]].sum(axis=1)
            counts = finite[rows[i]][:, columns[i]].sum(axis=1)
            means[i] = np.mean(sums / np.maximum(counts, 1))
    tail = (1.0 - level) / 2
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {"mean": float(np.nanmean(np.nanmean(matrix, axis=1))), "se": float(means.std(ddof=1)),
            "low": float(low), "high": float(high), "n": int(n)}


def paired_difference(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame, **options) -> dict[str, float]:
    """a - b on the sequences both have, with its bootstrap interval (options of `bootstrap`). Series are indexed by
    sequence; DataFrames have one row per seed and one column per sequence (b may be a Series, the same for every
    seed of a, as a baseline is)."""
    a_frame = a.to_frame().T if isinstance(a, pd.Series) else a
    b_frame = b.to_frame().T if isinstance(b, pd.Series) else b
    common = a_frame.columns.intersection(b_frame.columns)
    difference = a_frame[common].to_numpy(dtype=np.float64) - b_frame[common].to_numpy(dtype=np.float64)
    return bootstrap(difference, **options)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve of a score that should be higher for the positive cases: the probability that a
    random positive scores above a random negative, ties counted half (Mann-Whitney)."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    positive, negative = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([positive, negative])).rank().to_numpy()
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def bootstrap_auroc(positive: np.ndarray, negative: np.ndarray, paired: bool = True, level: float = LEVEL,
                    resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, float]:
    """AUROC with a percentile bootstrap interval. Paired cases (the same tuple gives one positive and one negative,
    in the same order) are resampled together; otherwise each group is resampled on its own."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if paired and len(positive) != len(negative):
        raise ValueError("paired cases need as many positives as negatives")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for i in range(resamples):
        if paired:
            pick = rng.integers(0, len(positive), len(positive))
            values[i] = auroc(positive[pick], negative[pick])
        else:
            values[i] = auroc(positive[rng.integers(0, len(positive), len(positive))],
                              negative[rng.integers(0, len(negative), len(negative))])
    tail = (1.0 - level) / 2
    low, high = np.nanquantile(values, [tail, 1.0 - tail])
    return {"auroc": auroc(positive, negative), "low": float(low), "high": float(high),
            "n": int(min(len(positive), len(negative)) if paired else len(positive) + len(negative))}
