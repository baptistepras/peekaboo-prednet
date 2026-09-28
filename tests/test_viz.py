"""Tests for contact sheets and GIFs."""

from pathlib import Path

import numpy as np
from PIL import Image

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import STATE_OCCLUDED
from peekaboo.data.render import RenderedSequence, RenderSettings, render_sequence
from peekaboo.paths import CONFIGS_DIR
from peekaboo.viz.frames import GAP, STRIP, animation_frames, contact_sheet, frame_tile, save_gif

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


def occlusion(pool: DigitPool, settings: GeneratorSettings) -> RenderedSequence:
    """Render one occlusion sequence."""
    spec = sample_spec(pool, settings, "test", 0, 0, condition="occlusion", k=6, speed=2)
    return render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)


def test_frame_tile_layout(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A tile stacks the observed frame, a gap, the amodal frame, and the state strip, all enlarged."""
    r = occlusion(pool, settings)
    h, w = r.spec.frame_height, r.spec.frame_width
    assert frame_tile(r, 0, scale=2).size == (2 * w, 2 * h + GAP + 2 * h + STRIP)
    assert frame_tile(r, 0, scale=2, show_amodal=False).size == (2 * w, 2 * h + STRIP)


def test_hidden_center_is_cyan(pool: DigitPool, settings: GeneratorSettings) -> None:
    """While the digit is hidden, the cross marking its true center is drawn in cyan over the bar."""
    r = occlusion(pool, settings)
    t = r.spec.onset_frame
    assert r.truth.state[t] == STATE_OCCLUDED
    tile = np.asarray(frame_tile(r, t, scale=3, show_amodal=False))
    cy, cx = r.truth.center[t]
    y, x = int((cy + 0.5) * 3), int((cx + 0.5) * 3)
    around = tile[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3)
    assert any(tuple(p) == (0, 255, 255) for p in around)


def test_contact_sheet_grid(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A sheet of every other frame of a 40 frame sequence, 10 per row, has 2 rows of tiles."""
    r = occlusion(pool, settings)
    sheet = contact_sheet(r, "title", step=2, scale=1, columns=10)
    tile_w, tile_h = frame_tile(r, 0, scale=1).size
    assert sheet.width == GAP + 10 * (tile_w + GAP)
    assert sheet.height > 2 * tile_h


def test_gif_has_one_frame_per_time_step(tmp_path: Path, pool: DigitPool, settings: GeneratorSettings) -> None:
    """The saved GIF has one image per frame of the sequence and loops."""
    r = occlusion(pool, settings)
    path = save_gif(animation_frames([r, r], ["left", "right"], scale=1), tmp_path / "demo.gif")
    with Image.open(path) as gif:
        assert gif.n_frames == r.spec.seq_len
        assert gif.info.get("loop") == 0
