"""Sequence conditions (control, occlusion, hidden bounce, blackout) and their conversion into SequenceSpecs.

- control: the bar is narrower than the digit, which passes behind it without ever being fully hidden (k = 0).
- occlusion: the digit crosses behind the bar and is fully hidden for exactly k frames.
- hidden_bounce: the bar stands against a wall; the digit bounces off the wall while hidden and comes back out.
- blackout: a control sequence whose frames are all set to zero for k frames at the most covered frame.
Surprise sequences are built from occlusion sequences in peekaboo.data.splicing.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import (FULLY_VISIBLE, CrossingSettings, OcclusionPlan, column_ink, plan_contact,
                                    plan_crossing, plan_hidden_bounce, visible_fraction)
from peekaboo.data.spec import SequenceSpec
from peekaboo.seeding import derive_seed

CONDITIONS = ("control", "occlusion", "hidden_bounce", "blackout")


@dataclass(frozen=True)
class GeneratorSettings:
    """All generator settings: placement rules plus condition mix and surprise parameters."""

    crossing: CrossingSettings
    x_speeds: tuple[int, ...]
    train_mix: dict[str, float]
    k_weights: dict[int, float]
    offset_px: int
    early_fraction: float
    max_digit_draws: int = 20
    tuple_attempts: int = 100  # placements of A tried per digit when building a surprise tuple

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeneratorSettings":
        """Build the settings from a data config dictionary."""
        conditions, surprises = config["conditions"], config["surprises"]
        return cls(crossing=CrossingSettings.from_config(config),
                   x_speeds=tuple(config["motion"]["x_speeds"]),
                   train_mix={str(name): float(p) for name, p in conditions["train_mix"].items()},
                   k_weights={int(k): float(w) for k, w in conditions["k_weights"].items()},
                   offset_px=int(surprises["offset_px"]), early_fraction=float(surprises["early_fraction"]),
                   max_digit_draws=int(conditions["max_digit_draws"]),
                   tuple_attempts=int(surprises.get("max_attempts", 100)))


def plan_condition(rng: np.random.Generator, sprite: np.ndarray, condition: str, k: int, speed: int,
                   settings: GeneratorSettings) -> tuple[OcclusionPlan, tuple[int, ...]]:
    """Place one sequence of the given condition. Returns the plan and the blackout frames (empty if none)."""
    cs = settings.crossing
    if condition == "occlusion":
        return plan_crossing(rng, sprite, k, cs, speeds=[speed]), ()
    if condition == "hidden_bounce":
        return plan_hidden_bounce(rng, sprite, k, cs, speeds=[speed]), ()
    if condition == "control":
        return plan_contact(rng, sprite, cs, speeds=[speed]), ()
    if condition == "blackout":
        # a control placement whose most covered frame starts k black frames, with room for the post frames and
        # no other contact with the bar inside the window, which may reach past the contact when k is long
        for _ in range(cs.max_attempts):
            plan = plan_contact(rng, sprite, cs, speeds=[speed])
            start, exit_frame = plan.event.onset_frame, plan.event.exit_frame
            window_end = max(exit_frame, start + k - 1) + cs.n_post
            if window_end <= cs.seq_len - 1 and (plan.fractions[exit_frame + 1:window_end + 1] >= FULLY_VISIBLE).all():
                return plan, tuple(range(start, start + k))
        raise RuntimeError(f"No blackout placement with k={k}, speed={speed}.")
    raise ValueError(f"Unknown condition '{condition}', expected one of {CONDITIONS}.")


def make_spec(pool: DigitPool, digit: int, identity: dict[str, Any], positions: np.ndarray, velocities: np.ndarray,
              bar_left: int, bar_width: int, present: np.ndarray | None = None, **fields: Any) -> SequenceSpec:
    """Assemble a SequenceSpec from its identity (split, index, seed), digit, motion, bar, and event fields."""
    seq_len = len(positions)
    return SequenceSpec(split=identity["split"], index=identity["index"], seed=identity["seed"],
                        frame_height=identity["frame_height"], frame_width=identity["frame_width"], seq_len=seq_len,
                        digit_index=digit, mnist_index=int(pool.mnist_indices[digit]), label=int(pool.labels[digit]),
                        ink_height=int(pool.heights[digit]), ink_width=int(pool.widths[digit]),
                        positions=np.asarray(positions, dtype=np.int64),
                        velocities=np.asarray(velocities, dtype=np.int64),
                        digit_present=np.ones(seq_len, dtype=bool) if present is None else np.asarray(present, bool),
                        bar_left=int(bar_left), bar_width=int(bar_width), **fields)


def spec_from_plan(pool: DigitPool, digit: int, identity: dict[str, Any], plan: OcclusionPlan, condition: str,
                   k: int, settings: GeneratorSettings, blackout: tuple[int, ...] = (), **fields: Any) -> SequenceSpec:
    """Turn a placement plan into a spec, filling the event frames and the analysis window."""
    e = plan.event
    cs = settings.crossing
    onset, reappear = e.onset_frame, e.reappear_frame
    if blackout:
        onset, reappear = blackout[0], blackout[-1] + 1
    return make_spec(pool, digit, identity, plan.trajectory.positions, plan.trajectory.velocities,
                     plan.bar_left, plan.bar_width, blackout_frames=tuple(blackout), condition=condition,
                     k_target=k, speed=plan.speed, entry_frame=e.entry_frame, onset_frame=onset,
                     expected_reappear_frame=reappear, actual_reappear_frame=reappear, exit_frame=e.exit_frame,
                     window_start=e.entry_frame - cs.n_context,
                     window_end=max(e.exit_frame, reappear - 1) + cs.n_post,
                     extra={"attempts": plan.attempts}, **fields)


def identity_of(split: str, index: int, seed: int, settings: GeneratorSettings) -> dict[str, Any]:
    """Return the identity fields shared by every spec of one sequence."""
    cs = settings.crossing
    return {"split": split, "index": index, "seed": seed, "frame_height": cs.frame_height,
            "frame_width": cs.frame_width}


def sample_spec(pool: DigitPool, settings: GeneratorSettings, split: str, index: int, base_seed: int,
                condition: str | None = None, k: int | None = None, speed: int | None = None) -> SequenceSpec:
    """Build sequence `index` of a split, fully determined by (base_seed, split, index) and the fixed arguments.

    Unset arguments are drawn: the condition from the training mix, k from the k weights, and the speed among the
    speeds allowed for k. If a digit cannot be placed, another digit is drawn.
    """
    seed = derive_seed(base_seed, split, index)
    rng = np.random.default_rng(seed)
    if condition is None:
        names = list(settings.train_mix)
        probs = np.array([settings.train_mix[n] for n in names])
        condition = str(rng.choice(names, p=probs / probs.sum()))
    if condition == "control":
        k = 0
        speed = speed if speed is not None else int(rng.choice(settings.x_speeds))
    else:
        if k is None:
            ks = list(settings.k_weights)
            weights = np.array([settings.k_weights[v] for v in ks])
            k = int(rng.choice(ks, p=weights / weights.sum()))
        speed = speed if speed is not None else int(rng.choice(settings.crossing.k_speeds[k]))

    identity = identity_of(split, index, seed, settings)
    for _ in range(settings.max_digit_draws):
        digit = pool.sample(rng)
        try:
            plan, blackout = plan_condition(rng, pool.sprite(digit), condition, k, speed, settings)
        except RuntimeError:
            continue
        return spec_from_plan(pool, digit, identity, plan, condition, k, settings, blackout)
    raise RuntimeError(f"Could not place a '{condition}' sequence with k={k}, speed={speed} "
                       f"after {settings.max_digit_draws} digits.")


def spec_visible_fraction(spec: SequenceSpec, sprite: np.ndarray) -> np.ndarray:
    """Return the fraction of the digit's ink that is visible on screen, per frame.

    It is 0 where the digit is absent (after a vanish) or during a blackout.
    """
    fractions = visible_fraction(column_ink(sprite), spec.positions[:, 1], spec.bar_left, spec.bar_width)
    fractions[~spec.digit_present] = 0.0
    fractions[list(spec.blackout_frames)] = 0.0
    return fractions
