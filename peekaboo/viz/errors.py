"""Error maps of PredNet (step 3.5): the activity of the error units of every layer, frame by frame, under the actual
and the predicted frames.

The map of a layer at frame t is the mean of its error units E_l[t] (positive and negative errors, both rectified)
over their channels, enlarged to the frame with nearest neighbor pixels, so each cell of a deep layer shows as a
block. Each layer of each sequence has its own scale: its MAP_QUANTILE quantile over the frames from 1 on is drawn
at full brightness, so the maps show where and when a layer's errors rise and fall, not how large they are against
another layer's.
"""

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from peekaboo.data.render import RenderedSequence
from peekaboo.data.truth import STATE_NAMES
from peekaboo.viz.frames import BLACK, GAP, LINE, STATE_RGB, STRIP, WHITE, font, upscale
from peekaboo.viz.predictions import LABEL_WIDTH, actual_image, text_width, to_uint8

COLORMAP = "magma"
MAP_QUANTILE = 0.99


def error_maps(errors: list[list[torch.Tensor]], frame_size: tuple[int, int]) -> np.ndarray:
    """(B, T, L, H, W) mean error activity of every layer over its channels, enlarged to the frame, from the "E" of
    a forward pass with return_states (one list of per layer tensors (B, C, H_l, W_l) per time step)."""
    layers = []
    for l in range(len(errors[0])):
        stacked = torch.stack([step[l] for step in errors], dim=1)  # (B, T, C, H_l, W_l)
        size, steps = stacked.shape[:2]
        mean = stacked.mean(dim=2).flatten(0, 1)[:, None]
        layers.append(F.interpolate(mean, size=frame_size, mode="nearest")[:, 0].reshape(size, steps, *frame_size))
    return torch.stack(layers, dim=2).float().cpu().numpy()


def normalize(maps: np.ndarray) -> np.ndarray:
    """The maps (T, L, H, W) of one sequence in [0, 1]: every layer divided by its MAP_QUANTILE quantile over the
    frames from 1 on (frame 0 is predicted before any input), and clipped."""
    scales = np.quantile(maps[1:].transpose(1, 0, 2, 3).reshape(maps.shape[1], -1), MAP_QUANTILE, axis=1)
    return np.clip(maps / np.maximum(scales, 1e-12)[None, :, None, None], 0.0, 1.0)


def colorize(values: np.ndarray) -> np.ndarray:
    """Values in [0, 1] (H, W) as (H, W, 3) uint8 pixels of the colormap."""
    return to_uint8(matplotlib.colormaps[COLORMAP](values)[..., :3])


def row_names(layers: int) -> list[str]:
    """The names of the rows of a tile."""
    return ["actual", "predicted"] + [f"E layer {l}" for l in range(layers)]


def row_tops(rendered: RenderedSequence, layers: int, scale: int) -> list[int]:
    """The top pixel of every row inside a tile: the actual frame and its state strip, the prediction, and the maps."""
    height = rendered.spec.frame_height * scale
    tops = [0, height + STRIP + GAP]
    for _ in range(layers):
        tops.append(tops[-1] + height + GAP)
    return tops


def error_tile(rendered: RenderedSequence, prediction: np.ndarray, maps: np.ndarray, t: int,
               scale: int = 2) -> Image.Image:
    """Frame t as images stacked: the actual frame (hidden ink in cyan) with its state strip, the prediction, and the
    normalized error map of every layer (maps (T, L, H, W) in [0, 1])."""
    width = rendered.spec.frame_width * scale
    gap = np.full((GAP, width, 3), 255, dtype=np.uint8)
    strip = np.full((STRIP, width, 3), STATE_RGB[int(rendered.truth.state[t])], dtype=np.uint8)
    parts = [upscale(actual_image(rendered, t), scale), strip, gap, upscale(to_uint8(prediction[t]), scale)]
    for l in range(maps.shape[1]):
        parts += [gap, upscale(colorize(maps[t, l]), scale)]
    return Image.fromarray(np.concatenate(parts, axis=0))


def draw_names(draw: ImageDraw.ImageDraw, rendered: RenderedSequence, layers: int, scale: int, top: int) -> None:
    """Write the row names in the left column, next to a tile whose top is at `top`."""
    height = rendered.spec.frame_height * scale
    for name, row_top in zip(row_names(layers), row_tops(rendered, layers, scale)):
        draw.text((GAP, top + row_top + height // 2), name, fill=BLACK, font=font(10), anchor="lm")


LEGEND = ("actual: observed frame, hidden ink in cyan; strip: state; predicted: prediction of frame t from frames 0 "
          "to t-1; E layer l: mean activity of the error units of layer l (dark: none, bright: the layer's "
          f"{100 * MAP_QUANTILE:g}th percentile over the sequence)")


def error_sheet(rendered: RenderedSequence, prediction: np.ndarray, maps: np.ndarray, times: list[int], title: str,
                scale: int = 2, columns: int = 10) -> Image.Image:
    """Tiles of the given frames in a grid, with the row names on the left, frame numbers, a title, and a legend."""
    tiles = [error_tile(rendered, prediction, maps, t, scale) for t in times]
    tile_w, tile_h = tiles[0].size
    columns = min(columns, len(tiles))
    rows = -(-len(tiles) // columns)
    header = 2 * LINE + GAP
    width = max(LABEL_WIDTH + columns * (tile_w + GAP), text_width((title,), 12) + 2 * GAP,
                text_width((LEGEND,), 10) + 2 * GAP)
    sheet = Image.new("RGB", (width, header + rows * (tile_h + LINE + GAP) + GAP), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, GAP), title, fill=BLACK, font=font(12))
    draw.text((GAP, GAP + LINE), LEGEND, fill=(90, 90, 90), font=font(10))
    for i, (t, tile) in enumerate(zip(times, tiles)):
        left = LABEL_WIDTH + (i % columns) * (tile_w + GAP)
        top = header + (i // columns) * (tile_h + LINE + GAP)
        if i % columns == 0:
            draw_names(draw, rendered, maps.shape[1], scale, top)
        sheet.paste(tile, (left, top))
        draw.text((left + tile_w // 2, top + tile_h + 1), f"t={t}", fill=BLACK, font=font(10), anchor="ma")
    return sheet


def error_animation(renders: list[RenderedSequence], predictions: list[np.ndarray], maps: list[np.ndarray],
                    labels: list[str], times: list[int], scale: int = 2) -> list[Image.Image]:
    """One image per frame in `times`, with the sequences side by side, each under its label and its current state."""
    images = []
    for t in times:
        tiles = [error_tile(r, p, m, t, scale) for r, p, m in zip(renders, predictions, maps)]
        tile_w, tile_h = tiles[0].size
        top = 2 * LINE + GAP
        image = Image.new("RGB", (LABEL_WIDTH + len(tiles) * (tile_w + GAP), top + tile_h + GAP), WHITE)
        draw = ImageDraw.Draw(image)
        draw_names(draw, renders[0], maps[0].shape[1], scale, top)
        for i, (r, label, tile) in enumerate(zip(renders, labels, tiles)):
            left = LABEL_WIDTH + i * (tile_w + GAP)
            state = int(r.truth.state[t])
            draw.text((left, GAP), label, fill=BLACK, font=font(11))
            draw.text((left, GAP + LINE), f"t={t}  {STATE_NAMES[state]}", fill=STATE_RGB[state], font=font(11))
            image.paste(tile, (left, top))
        images.append(image)
    return images
