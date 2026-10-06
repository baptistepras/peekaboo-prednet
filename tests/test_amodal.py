"""Tests for the amodal decoder: its minimal form, learning, saving, the scores, the template baselines, the
evaluation with its controls, and the magenta digit in the figures."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, render_sequence
from peekaboo.data.truth import STATE_NAMES
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR
from peekaboo.probes.amodal import (AmodalDecoder, decoder_loss, evaluate_decoders, fit_decoder, ink_scores, shift,
                                    template_images)
from peekaboo.trackers.baselines import run_trackers
from peekaboo.trackers.detector import detect
from peekaboo.viz.predictions import MAGENTA, blend_magenta, prediction_sheet

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")
TINY = {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"}


def test_minimal_decoder_is_a_1x1_convolution_of_the_stacked_states() -> None:
    """With a 1 x 1 kernel, upsampling each layer's output equals one 1 x 1 convolution of all the layers upsampled
    and stacked."""
    torch.manual_seed(0)
    decoder = AmodalDecoder((2, 3), (8, 12))
    states = [torch.rand(4, 2, 8, 12), torch.rand(4, 3, 4, 6)]
    stacked = torch.cat([states[0], F.interpolate(states[1], size=(8, 12), mode="bilinear", align_corners=False)], 1)
    weight = torch.cat([conv.weight for conv in decoder.convs], dim=1)
    expected = F.conv2d(stacked, weight) + decoder.bias.view(1, 1, 1, 1)
    assert torch.allclose(decoder(states), expected, atol=1e-6)
    assert sum(p.numel() for p in decoder.parameters()) == 2 + 3 + 1


def test_decoder_learns_an_easy_mapping_and_reloads(tmp_path: Path) -> None:
    """When one state channel holds the digit, the decoder learns to draw it (IoU above 0.9), and the saved decoder
    decodes exactly as the trained one."""
    torch.manual_seed(0)
    target = (torch.rand(32, 8, 12) < 0.2).float()
    states = [torch.stack([4 * target - 2, torch.randn(32, 8, 12)], dim=1), torch.randn(32, 3, 4, 6)]
    decoder = AmodalDecoder((2, 3), (8, 12))
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.1)
    for _ in range(200):
        loss = decoder_loss(decoder, states, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, iou = ink_scores(decoder.imagine(states).detach(), target)
    assert iou.mean() > 0.9
    loaded = AmodalDecoder.load(decoder.save(tmp_path / "decoder.npz"))
    assert torch.equal(loaded.imagine(states), decoder.imagine(states))


def test_scores() -> None:
    """Identical images score 1, disjoint ink scores an IoU of 0, and undefined scores are NaN."""
    a = torch.zeros(4, 8, 12)
    a[:, 2:5, 3:6] = 1.0
    b = torch.zeros(4, 8, 12)
    b[:, 5:7, 8:10] = 1.0
    corr, iou = ink_scores(a, a)
    assert torch.allclose(corr, torch.ones(4)) and torch.allclose(iou, torch.ones(4))
    assert torch.equal(ink_scores(a, b)[1], torch.zeros(4))
    empty = torch.zeros(4, 8, 12)
    corr, iou = ink_scores(empty, empty)
    assert torch.isnan(corr).all() and torch.isnan(iou).all()


def test_shift_moves_and_fills_with_zeros() -> None:
    """A shifted image moves by whole pixels, and what comes in from the border is empty."""
    image = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(shift(image, 1, -1), [[0, 0, 0, 0], [1, 2, 3, 0], [5, 6, 7, 0]])
    assert np.array_equal(shift(image, 0, 0), image)


def test_template_with_walls_is_the_true_hidden_digit(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Moved by the exact tracker, the last fully visible digit is exactly the amodal digit while it is hidden."""
    for index in range(4):
        spec = sample_spec(pool, settings, "test", index, 0, condition="occlusion", k=6, speed=3)
        r = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
        frames = r.observed.transpose(0, 3, 1, 2)
        tracks = run_trackers(frames, r.truth.center, r.truth.state)
        templates = template_images(frames, detect(frames), tracks["kalman_walls"].prediction)
        hidden = np.flatnonzero(r.truth.state == STATE_NAMES.index("occluded"))
        assert hidden.size > 0
        assert np.allclose(templates[hidden], r.amodal[hidden] / 255.0, atol=1e-6)


def test_fitting_and_evaluation_run_with_their_controls(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A decoder on a model and one on a random model train on the stream and are evaluated next to the templates:
    one row per frame, scores in range, and the exact template perfect on hidden frames."""
    torch.manual_seed(0)
    stream = OnTheFlyDataset(pool, settings, RENDER, "decoder", 0, include_amodal=True)
    held_out = OnTheFlyDataset(pool, settings, RENDER, "val", 0, include_amodal=True)
    decoders = {}
    for name in ("decoder", "random_decoder"):
        model = build_model(TINY)
        decoder, losses = fit_decoder(model, stream, 2, CPU, batch_size=2, log=lambda line: None)
        assert len(losses) == 2 and all(np.isfinite(losses))
        decoders[name] = (model, decoder)
    table = evaluate_decoders(decoders, held_out, 3, CPU, batch_size=3)
    assert len(table) > 0 and (table["t"] >= 2).all()
    for column in ("decoder_iou", "random_decoder_corr", "template_kalman_iou", "template_walls_corr"):
        assert column in table
    assert table["decoder_corr"].dropna().between(-1, 1).all()
    hidden = table[(table["state"] == STATE_NAMES.index("occluded")) & table["template_walls_iou"].notna()]
    assert np.allclose(hidden["template_walls_iou"], 1.0)


def test_magenta_digit_in_the_figure(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The decoded intensity blends the prediction toward magenta, nothing where it is 0, and a sheet with both
    readouts renders."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    blended = blend_magenta(image, np.array([[1.0, 0.0], [np.nan, 0.5]]))
    assert tuple(blended[0, 0]) == tuple(int(round(0.8 * c)) for c in MAGENTA)
    assert tuple(blended[0, 1]) == (0, 0, 0) and tuple(blended[1, 0]) == (0, 0, 0)
    spec = sample_spec(pool, settings, "test", 0, 0, condition="occlusion", k=4, speed=3)
    rendered = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
    imagined = rendered.amodal / 255.0
    sheet = prediction_sheet(rendered, rendered.observed / 255.0, [5, 6], "test", belief=rendered.truth.center,
                             imagined=imagined)
    assert sheet.size[0] > 0
