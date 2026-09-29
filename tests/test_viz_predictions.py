"""Tests for the prediction figures: cyan exactly on hidden ink, outline outside the digit, and error colors."""

import numpy as np

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import STATE_OCCLUDED, STATE_VISIBLE
from peekaboo.data.render import RenderedSequence, RenderSettings, render_sequence
from peekaboo.paths import CONFIGS_DIR
from peekaboo.viz.frames import GAP, STRIP, upscale
from peekaboo.viz.predictions import (CYAN, WHITE, actual_image, dilate, error_image, frames_around_event,
                                      predicted_image, prediction_animation, prediction_sheet, prediction_tile)

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


def render(pool: DigitPool, settings: GeneratorSettings, condition: str = "occlusion") -> RenderedSequence:
    """Render one sequence of the given condition, with k = 6 at speed 2."""
    spec = sample_spec(pool, settings, "test", 0, 0, condition=condition, k=6, speed=2)
    return render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)


def perfect(rendered: RenderedSequence) -> np.ndarray:
    """A prediction equal to the observed frames, in [0, 1]."""
    return rendered.observed / 255.0


def test_actual_row_marks_exactly_the_hidden_ink(pool: DigitPool, settings: GeneratorSettings) -> None:
    """On every frame, only hidden ink pixels change, and they turn cyan (green and blue above red)."""
    for condition in ("occlusion", "blackout"):
        r = render(pool, settings, condition)
        assert (r.truth.state == STATE_OCCLUDED).any() or r.spec.blackout_frames
        for t in range(r.spec.seq_len):
            image = actual_image(r, t)
            changed = np.any(image != r.observed[t], axis=2)
            hidden = (r.amodal[t] > 0) & ~r.modal_mask[t]
            assert np.array_equal(changed, hidden), (condition, t)
            pixels = image[hidden].astype(int)
            assert np.all(pixels[:, 1] > pixels[:, 0]) and np.all(pixels[:, 2] > pixels[:, 0])


def test_predicted_row_outline(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The outline lies just outside the true digit; it is cyan when hidden and white when fully visible."""
    r = render(pool, settings)
    scale = 3
    for t, color in ((r.spec.onset_frame, CYAN), (r.spec.entry_frame - 1, WHITE)):
        image = predicted_image(r, perfect(r), t, scale)
        ink = upscale(r.amodal[t] > 0, scale)
        changed = np.any(image != upscale(r.observed[t], scale), axis=2)
        assert changed.any()
        assert np.array_equal(changed, dilate(ink) & ~ink)
        assert np.all(image[changed] == color), t
    assert r.truth.state[r.spec.entry_frame - 1] == STATE_VISIBLE


def test_error_row_colors(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A perfect prediction has no error; a black prediction is all red, a white one all blue."""
    r = render(pool, settings)
    t = r.spec.onset_frame
    assert not error_image(r, perfect(r), t).any()
    missed = error_image(r, np.zeros_like(perfect(r)), t).astype(int)
    lit = missed.any(axis=2)
    assert np.array_equal(lit, r.observed[t].any(axis=2))
    assert np.all(missed[lit][:, 0] > missed[lit][:, 2])
    extra = error_image(r, np.ones_like(perfect(r)), t).astype(int)
    assert np.all(extra[..., 2] > extra[..., 0])


def test_tile_sheet_and_animation_sizes(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A tile stacks three enlarged frames, the sheet holds every tile, and the animation has one image per frame."""
    r = render(pool, settings)
    h, w = r.spec.frame_height, r.spec.frame_width
    times = frames_around_event(r.spec)
    assert times[0] >= 1 and times[0] < r.spec.entry_frame and times[-1] > r.spec.exit_frame
    tile = prediction_tile(r, perfect(r), times[0], scale=2)
    assert tile.size == (2 * w, 3 * 2 * h + STRIP + 2 * GAP)
    sheet = prediction_sheet(r, perfect(r), times, "title", scale=2, columns=5)
    rows = -(-len(times) // 5)
    assert sheet.width >= 5 * tile.width and sheet.height > rows * tile.height
    images = prediction_animation([r, r], [perfect(r)] * 2, ["a", "b"], times, scale=1)
    assert len(images) == len(times)
