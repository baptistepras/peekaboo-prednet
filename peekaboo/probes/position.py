"""Position probes (option B, decision D15): read the digit's position from a frozen model's state.

Two probes, trained on frames of the training stream and evaluated on held out sequences:
- a ridge regression of the centroid (y, x) on the standardized features. y is the main axis while the digit is hidden
  (critique C4): the bar spans the full height, so x is bounded by the bar and partly known from it, while y keeps
  changing behind it;
- a logistic regression per axis on 8 position bins, which gives a probability for each bin, so the probe's
  uncertainty can be read and shown.
The regularization of each probe is chosen on a validation split of the training sequences (whole sequences, so no
frame of a validation sequence is seen in training). Probes are saved as plain arrays (.npz), never as pickles.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

AXES = ("y", "x")
BINS = 8
RIDGE_ALPHAS = tuple(10.0 ** p for p in range(-1, 6))
LOGISTIC_C = (1e-3, 1e-2, 1e-1, 1.0)


@dataclass(frozen=True)
class Standardizer:
    """Mean and scale of each feature, from the training frames."""

    mean: np.ndarray   # (F,)
    scale: np.ndarray  # (F,), 1 for constant features

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        """Mean and standard deviation of each feature."""
        mean = features.mean(axis=0, dtype=np.float64)
        scale = features.std(axis=0, dtype=np.float64)
        return cls(mean=mean, scale=np.where(scale > 1e-12, scale, 1.0))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Standardized features, float32."""
        return ((features - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class PositionProbe:
    """A fitted ridge probe of (y, x) and a logistic probe of the position bins of each axis."""

    standardizer: Standardizer
    coef: np.ndarray            # (F, 2) ridge weights for y and x
    intercept: np.ndarray       # (2,)
    alpha: float                # chosen ridge penalty
    bin_edges: np.ndarray       # (2, BINS + 1) bin edges in pixels for y and x
    bin_coef: np.ndarray        # (2, BINS, F) logistic weights per axis and bin
    bin_intercept: np.ndarray   # (2, BINS)
    c: float                    # chosen logistic inverse penalty
    layer_limit: int            # pooling limit of the features it reads (see peekaboo/probes/features.py)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """(N, 2) predicted centroid (y, x) in pixels."""
        return self.standardizer(features) @ self.coef.astype(np.float32) + self.intercept

    def bin_probabilities(self, features: np.ndarray) -> np.ndarray:
        """(N, 2, BINS) probability of each bin, for y and x."""
        z = self.standardizer(features)
        logits = np.einsum("nf,abf->nab", z, self.bin_coef.astype(np.float32)) + self.bin_intercept
        logits -= logits.max(axis=2, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=2, keepdims=True)

    def save(self, path: str | Path) -> Path:
        """Save as .npz, plain arrays only."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.standardizer.mean, scale=self.standardizer.scale, coef=self.coef,
                 intercept=self.intercept, alpha=self.alpha, bin_edges=self.bin_edges, bin_coef=self.bin_coef,
                 bin_intercept=self.bin_intercept, c=self.c, layer_limit=self.layer_limit)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PositionProbe":
        """Load a probe saved by save."""
        with np.load(path, allow_pickle=False) as data:
            return cls(standardizer=Standardizer(data["mean"], data["scale"]), coef=data["coef"],
                       intercept=data["intercept"], alpha=float(data["alpha"]), bin_edges=data["bin_edges"],
                       bin_coef=data["bin_coef"], bin_intercept=data["bin_intercept"], c=float(data["c"]),
                       layer_limit=int(data["layer_limit"]))


def position_bins(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(N, 2) bin index of each coordinate, for bin edges (2, BINS + 1)."""
    return np.stack([np.clip(np.searchsorted(edges[a], positions[:, a], side="right") - 1, 0, BINS - 1)
                     for a in range(2)], axis=1)


def default_bin_edges(frame_size: tuple[int, int]) -> np.ndarray:
    """(2, BINS + 1) equal bins over the frame height and width, in pixels."""
    return np.stack([np.linspace(-0.5, size - 0.5, BINS + 1) for size in frame_size])


def fit_ridge(z: np.ndarray, targets: np.ndarray, train: np.ndarray,
              alphas: tuple[float, ...] = RIDGE_ALPHAS) -> tuple[np.ndarray, np.ndarray, float, dict[float, float]]:
    """Ridge regression on standardized features z (N, F) for targets (N, K), fitted on the `train` frames for every
    penalty at once (one eigendecomposition), the penalty chosen by the mean squared error on the other frames.
    Returns the weights (F, K), the intercept (K,), the penalty, and the validation error of each penalty."""
    x = z[train].astype(np.float64)
    y_mean = targets[train].mean(axis=0)
    gram = x.T @ x
    eigenvalues, vectors = np.linalg.eigh(gram)
    projected = vectors.T @ (x.T @ (targets[train] - y_mean))
    held_out = z[~train].astype(np.float64)
    errors, weights = {}, {}
    for alpha in alphas:
        w = vectors @ (projected / (eigenvalues + alpha)[:, None])
        weights[alpha] = w
        errors[alpha] = float(((held_out @ w + y_mean - targets[~train]) ** 2).mean())
    best = min(errors, key=errors.get)
    return weights[best], y_mean, best, errors


def fit_bins(z: np.ndarray, bins: np.ndarray, train: np.ndarray,
             cs: tuple[float, ...] = LOGISTIC_C) -> tuple[np.ndarray, np.ndarray, float, dict[float, float]]:
    """Logistic regression per axis on standardized features z (N, F) for bin labels (N, 2), the inverse penalty
    chosen by the log loss on the frames outside `train`. Returns the weights (2, BINS, F), the intercepts (2, BINS),
    the inverse penalty, and the validation log loss of each value."""
    losses, fitted = {}, {}
    for c in cs:
        coef, intercept, loss = np.zeros((2, BINS, z.shape[1])), np.full((2, BINS), -1e9), 0.0  # -1e9: bin never seen
        for a in range(2):
            model = LogisticRegression(C=c, max_iter=300)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)  # 300 iterations at most, enough for a readout
                model.fit(z[train], bins[train, a])
            coef[a, model.classes_], intercept[a, model.classes_] = model.coef_, model.intercept_
            p = np.zeros((int((~train).sum()), BINS))
            p[:, model.classes_] = model.predict_proba(z[~train])
            right = p[np.arange(len(p)), bins[~train, a]]
            loss += float(-np.log(np.clip(right, 1e-12, 1.0)).mean())
        losses[c], fitted[c] = loss / 2, (coef, intercept)
    best = min(losses, key=losses.get)
    return fitted[best][0], fitted[best][1], best, losses


def fit_position_probe(features: np.ndarray, positions: np.ndarray, sequences: np.ndarray,
                       frame_size: tuple[int, int], layer_limit: int, validation_share: float = 0.2,
                       seed: int = 0) -> tuple[PositionProbe, dict[str, dict[float, float]]]:
    """Fit both probes on frames (N, F) with true centroids (N, 2) and the sequence of each frame. A share of the
    sequences is held out to choose the penalties. Returns the probe and the validation scores of every penalty."""
    rng = np.random.default_rng(seed)
    names = np.unique(sequences)
    held = rng.choice(names, size=max(1, int(round(validation_share * len(names)))), replace=False)
    train = ~np.isin(sequences, held)
    standardizer = Standardizer.fit(features[train])
    z = standardizer(features)
    coef, intercept, alpha, ridge_errors = fit_ridge(z, positions.astype(np.float64), train)
    edges = default_bin_edges(frame_size)
    bin_coef, bin_intercept, c, bin_losses = fit_bins(z, position_bins(positions, edges), train)
    probe = PositionProbe(standardizer=standardizer, coef=coef, intercept=intercept, alpha=alpha, bin_edges=edges,
                          bin_coef=bin_coef, bin_intercept=bin_intercept, c=c, layer_limit=layer_limit)
    return probe, {"ridge_mse": ridge_errors, "bins_log_loss": bin_losses}
