"""Surprise tuples in the style of PLATO (Piloto et al., 2022): two possible sequences and two impossible splices.

Possible A is an ordinary occlusion. Possible B is another plausible sequence with the same digit and bar, built
from A's state at a splice frame where the digit is fully hidden in both. Since a hidden digit is not drawn, frames
at the splice are identical, so the impossible sequences AB (A's start, B's end) and BA (B's start, A's end) show no
visible cut. The surprise signal is the error on AB after the splice minus the error on B, on identical frames.
"""

from dataclasses import dataclass

import numpy as np

from peekaboo.data.conditions import GeneratorSettings, identity_of, make_spec, spec_from_plan
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import (CrossingEvent, OcclusionPlan, column_ink, find_crossing, plan_crossing,
                                    visible_fraction, windows_ok)
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.trajectory import Bounds, Trajectory, build_trajectory
from peekaboo.seeding import derive_seed

SURPRISES = ("direction", "speed_fast", "speed_slow", "offset", "early", "vanish")


@dataclass(frozen=True)
class SurpriseTuple:
    """The four sequences of one surprise, sharing digit, bar, and splice frame."""

    kind: str
    splice_frame: int
    possible_a: SequenceSpec
    possible_b: SequenceSpec
    impossible_ab: SequenceSpec
    impossible_ba: SequenceSpec

    def specs(self) -> list[SequenceSpec]:
        """Return the four specs in the order A, B, AB, BA."""
        return [self.possible_a, self.possible_b, self.impossible_ab, self.impossible_ba]


def surprise_speeds(kind: str, settings: GeneratorSettings) -> list[int]:
    """Return the speeds of A allowed for a surprise: B's speed must also be a training speed.

    B is the matched control of the surprise, so it must not move at a speed the models never saw in training.
    This leaves 2 -> 4 px/frame for speed_fast and 4 -> 2 px/frame for speed_slow with the base config.
    """
    speeds = list(settings.x_speeds)
    if kind == "speed_fast":
        return [v for v in speeds if 2 * v in speeds]
    if kind == "speed_slow":
        return [v for v in speeds if v % 2 == 0 and v // 2 in speeds]
    return speeds


def early_jump_frames(k: int, settings: GeneratorSettings) -> int:
    """Number of frames an "early" surprise skips while hidden."""
    return max(1, int(round(settings.early_fraction * k)))


def surprise_state(kind: str, pos: tuple[int, int], vel: tuple[int, int], k: int, bounds: Bounds,
                   settings: GeneratorSettings,
                   rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return B's (position, velocity) at the splice frame, derived from A's, or None if the change is not possible."""
    (y, x), (vy, vx) = pos, vel
    if kind == "direction":
        return (y, x), (vy, -vx)
    if kind == "speed_fast":
        return (y, x), (vy, 2 * vx)
    if kind == "speed_slow":
        return None if vx % 2 else ((y, x), (vy, vx // 2))
    if kind == "offset":
        shifts = [d for d in (settings.offset_px, -settings.offset_px) if bounds.y_min <= y + d <= bounds.y_max]
        return ((y + int(rng.choice(shifts)), x), (vy, vx)) if shifts else None
    if kind == "early":
        return (y, x + int(np.sign(vx)) * early_jump_frames(k, settings) * abs(vx)), (vy, vx)
    raise ValueError(f"Unknown surprise '{kind}', expected one of {SURPRISES}.")


def changes_reappearance(kind: str, event_a: CrossingEvent, event_b: CrossingEvent) -> bool:
    """True if a timing surprise really moves the reappearance: earlier for speed_fast and early, later for speed_slow.

    A change made on the last hidden frames can leave the reappearance frame unchanged, which would make it invisible.
    """
    if kind in ("speed_fast", "early"):
        return event_b.reappear_frame < event_a.reappear_frame
    if kind == "speed_slow":
        return event_b.reappear_frame > event_a.reappear_frame
    return True


def make_surprise_tuple(rng: np.random.Generator, pool: DigitPool, digit: int, kind: str, k: int, speed: int,
                        settings: GeneratorSettings, identity: dict, tuple_id: int) -> SurpriseTuple:
    """Build possible A, possible B, and the impossible splices AB and BA for one digit.

    The change happens strictly while the digit is hidden: A and B are both fully occluded at the frame before
    the splice and at the splice frame, and timing surprises must move the reappearance. Raises RuntimeError if no
    valid tuple is found.
    """
    if kind not in SURPRISES:
        raise ValueError(f"Unknown surprise '{kind}', expected one of {SURPRISES}.")
    allowed = surprise_speeds(kind, settings)
    if speed not in allowed:
        raise ValueError(f"'{kind}' is only built from speeds {allowed}, so that B keeps a training speed.")
    jump = early_jump_frames(k, settings) if kind == "early" else 0
    if k < jump + 2:
        raise ValueError(f"'{kind}' needs k >= {jump + 2} so the change can happen strictly while hidden.")
    cs = settings.crossing
    sprite = pool.sprite(digit)
    col_ink = column_ink(sprite)
    bounds = Bounds.for_sprite(cs.frame_height, cs.frame_width, *sprite.shape)

    for _ in range(settings.tuple_attempts):
        plan_a = plan_crossing(rng, sprite, k, cs, speeds=[speed])
        ea = plan_a.event
        first, last = ea.onset_frame + 1, ea.reappear_frame - 1 - jump
        if first > last:
            continue
        t_s = int(rng.integers(first, last + 1))

        plan_b = None
        if kind != "vanish":
            state = surprise_state(kind, tuple(plan_a.trajectory.positions[t_s]),
                                   tuple(plan_a.trajectory.velocities[t_s]), k, bounds, settings, rng)
            if state is None:
                continue
            try:
                traj_b = build_trajectory(t_s, state[0], state[1], cs.seq_len, bounds)
            except ValueError:
                continue
            fractions_b = visible_fraction(col_ink, traj_b.positions[:, 1], plan_a.bar_left, plan_a.bar_width)
            hidden = fractions_b <= cs.thresholds.occluded_max
            if not (hidden[t_s - 1] and hidden[t_s]):
                continue
            eb = find_crossing(fractions_b, cs.thresholds, frame=t_s)
            if eb is None or not windows_ok(traj_b, fractions_b, eb, cs):
                continue
            if not changes_reappearance(kind, ea, eb):
                continue
            plan_b = (traj_b, eb)
        return _assemble(pool, digit, kind, k, plan_a, plan_b, t_s, settings, identity, tuple_id)
    raise RuntimeError(f"No valid '{kind}' tuple for k={k}, speed={speed} after {settings.tuple_attempts} attempts.")


def _assemble(pool: DigitPool, digit: int, kind: str, k: int, plan_a: OcclusionPlan,
              plan_b: tuple[Trajectory, CrossingEvent] | None, t_s: int, settings: GeneratorSettings,
              identity: dict, tuple_id: int) -> SurpriseTuple:
    """Build the four specs of a tuple from A's plan, B's trajectory and event (None for vanish), and the splice."""
    cs = settings.crossing
    ea: CrossingEvent = plan_a.event
    ta = plan_a.trajectory
    bar = (plan_a.bar_left, plan_a.bar_width)
    common = {"surprise_frame": t_s, "tuple_id": tuple_id}
    a_window = (ea.entry_frame - cs.n_context, ea.exit_frame + cs.n_post)

    spec_a = spec_from_plan(pool, digit, identity, plan_a, "occlusion", k, settings, tuple_role="A", **common)
    seq_len = len(ta.positions)
    if plan_b is None:
        # vanish: B is the same scene with no digit at all
        absent = np.zeros(seq_len, dtype=bool)
        spec_b = make_spec(pool, digit, identity, ta.positions, ta.velocities, *bar, present=absent,
                           condition="empty", window_start=a_window[0], window_end=a_window[1],
                           tuple_role="B", **common)
        before = np.arange(seq_len) < t_s
        spec_ab = make_spec(pool, digit, identity, ta.positions, ta.velocities, *bar, present=before,
                            condition="surprise", surprise_type=kind, k_target=k, speed=plan_a.speed,
                            entry_frame=ea.entry_frame, onset_frame=ea.onset_frame,
                            expected_reappear_frame=ea.reappear_frame, window_start=a_window[0],
                            window_end=a_window[1], tuple_role="AB", **common)
        spec_ba = make_spec(pool, digit, identity, ta.positions, ta.velocities, *bar, present=~before,
                            condition="surprise", surprise_type="appear", k_target=k, speed=plan_a.speed,
                            actual_reappear_frame=ea.reappear_frame, exit_frame=ea.exit_frame,
                            window_start=a_window[0], window_end=a_window[1], tuple_role="BA", **common)
        return SurpriseTuple(kind, t_s, spec_a, spec_b, spec_ab, spec_ba)

    tb, eb = plan_b
    speed_b = int(abs(tb.velocities[t_s, 1]))
    spec_b = make_spec(pool, digit, identity, tb.positions, tb.velocities, *bar, condition="occlusion",
                       k_target=eb.k, speed=speed_b, entry_frame=eb.entry_frame, onset_frame=eb.onset_frame,
                       expected_reappear_frame=eb.reappear_frame, actual_reappear_frame=eb.reappear_frame,
                       exit_frame=eb.exit_frame, window_start=eb.entry_frame - cs.n_context,
                       window_end=eb.exit_frame + cs.n_post, tuple_role="B", **common)

    def splice(first: Trajectory, first_event: CrossingEvent, second: Trajectory, second_event: CrossingEvent,
               role: str, speed_before: int, speed_after: int) -> SequenceSpec:
        """Join the start of one trajectory with the end of the other at the splice frame."""
        positions = np.concatenate([first.positions[:t_s], second.positions[t_s:]])
        velocities = np.concatenate([first.velocities[:t_s], second.velocities[t_s:]])
        return make_spec(pool, digit, identity, positions, velocities, *bar, condition="surprise",
                         surprise_type=kind, k_target=k, speed=speed_before,
                         entry_frame=first_event.entry_frame, onset_frame=first_event.onset_frame,
                         expected_reappear_frame=first_event.reappear_frame,
                         actual_reappear_frame=second_event.reappear_frame, exit_frame=second_event.exit_frame,
                         window_start=first_event.entry_frame - cs.n_context,
                         window_end=second_event.exit_frame + cs.n_post, tuple_role=role,
                         extra={"speed_after": speed_after}, **common)

    spec_ab = splice(ta, ea, tb, eb, "AB", plan_a.speed, speed_b)
    spec_ba = splice(tb, eb, ta, ea, "BA", speed_b, plan_a.speed)
    return SurpriseTuple(kind, t_s, spec_a, spec_b, spec_ab, spec_ba)


def sample_surprise_tuple(pool: DigitPool, settings: GeneratorSettings, split: str, index: int, base_seed: int,
                          kind: str, k: int, speed: int) -> SurpriseTuple:
    """Build surprise tuple `index`, fully determined by (base_seed, split, index, kind), drawing new digits on failure."""
    seed = derive_seed(base_seed, split, "surprise", kind, index)
    rng = np.random.default_rng(seed)
    identity = identity_of(split, index, seed, settings)
    for _ in range(settings.max_digit_draws):
        digit = pool.sample(rng)
        try:
            return make_surprise_tuple(rng, pool, digit, kind, k, speed, settings, identity, tuple_id=index)
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not build a '{kind}' tuple with k={k}, speed={speed} "
                       f"after {settings.max_digit_draws} digits.")
