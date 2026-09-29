"""Figures of a model's predictions next to the truth, drawn with Pillow so every pixel stays sharp.

Each frame t becomes a tile of three images:
- actual: the observed frame t, with the hidden ink of the digit blended toward cyan (from the amodal frame), and a
  strip colored by state below it;
- predicted: the model's prediction of frame t from frames 0 to t - 1, with a one pixel outline around the true
  digit, cyan next to hidden ink and white next to visible ink;
- error: PredNet's pixel error units E_0, red where the frame is brighter than the prediction (something missed),
  blue where the prediction is brighter than the frame (something predicted that is not there).

The prediction cannot show a hidden digit: behind the bar, the correct prediction of the next frame is the gray bar.
What the model believes about the hidden digit lives in its internal states, and is read by the probes of later steps.
"""

import numpy as np
from PIL import Image, ImageDraw

from peekaboo.data.render import RenderedSequence
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.truth import STATE_NAMES
from peekaboo.viz.frames import BLACK, GAP, LINE, STATE_RGB, STRIP, WHITE, font, upscale

CYAN = (0, 255, 255)
ERROR_MISSED = (255, 40, 40)  # in the frame, not predicted
ERROR_EXTRA = (60, 120, 255)  # predicted, not in the frame
HIDDEN_ALPHA = 0.6            # how far hidden ink is blended toward cyan
ROW_NAMES = ("actual", "predicted", "error")
LABEL_WIDTH = 64              # left column with the row names


def legend(gain: float) -> tuple[str, str]:
    """Two lines explaining the rows and colors."""
    return ("actual: frame t, hidden ink in cyan; strip: green visible, orange partial, red occluded, gray blackout, "
            "dark absent",
            "predicted: prediction of frame t from frames 0 to t-1, outline of the true digit (cyan where hidden); "
            f"error (x{gain:g}): red = in the frame but not predicted, blue = predicted but not in the frame")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert values in [0, 1] to uint8 pixels."""
    return np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a 2D mask by one pixel up, down, left, and right."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def hidden_ink(rendered: RenderedSequence, t: int) -> np.ndarray:
    """Ink pixels of frame t that are not on screen (under the bar or blacked out)."""
    return (rendered.amodal[t] > 0) & ~rendered.modal_mask[t]


def actual_image(rendered: RenderedSequence, t: int, alpha: float = HIDDEN_ALPHA) -> np.ndarray:
    """Observed frame t (H, W, 3) uint8, with every hidden ink pixel blended toward cyan, more for brighter ink."""
    image = rendered.observed[t].astype(np.float64)
    hidden = hidden_ink(rendered, t)
    # faint ink still gets half the blend, so the whole hidden digit shows
    weight = alpha * (0.5 + 0.5 * rendered.amodal[t][hidden][:, None] / 255.0)
    image[hidden] = (1.0 - weight) * image[hidden] + weight * np.array(CYAN, dtype=np.float64)
    return np.rint(image).astype(np.uint8)


def predicted_image(rendered: RenderedSequence, prediction: np.ndarray, t: int, scale: int) -> np.ndarray:
    """Prediction of frame t, enlarged, with a one pixel outline just outside the true digit (inside is untouched)."""
    image = upscale(to_uint8(prediction[t]), scale)
    ink = upscale(rendered.amodal[t] > 0, scale)
    outline = dilate(ink) & ~ink
    image[outline] = WHITE
    image[outline & dilate(upscale(hidden_ink(rendered, t), scale))] = CYAN
    return image


def error_image(rendered: RenderedSequence, prediction: np.ndarray, t: int, gain: float = 2.0) -> np.ndarray:
    """PredNet's pixel errors for frame t (H, W, 3) uint8: positive part in red, negative part in blue, times gain."""
    frame = rendered.observed[t] / 255.0
    missed = np.clip(frame - prediction[t], 0.0, None).max(axis=2)
    extra = np.clip(prediction[t] - frame, 0.0, None).max(axis=2)
    colors = gain * (missed[..., None] * np.array(ERROR_MISSED) + extra[..., None] * np.array(ERROR_EXTRA)) / 255.0
    return to_uint8(colors)


def row_tops(rendered: RenderedSequence, scale: int) -> tuple[int, int, int]:
    """Top pixel of the actual, predicted, and error images inside a tile."""
    height = rendered.spec.frame_height * scale
    return 0, height + STRIP + GAP, 2 * height + STRIP + 2 * GAP


def prediction_tile(rendered: RenderedSequence, prediction: np.ndarray, t: int, scale: int = 2,
                    gain: float = 2.0) -> Image.Image:
    """Frame t as three images stacked: actual with its state strip, predicted, and error."""
    width = rendered.spec.frame_width * scale
    gap = np.full((GAP, width, 3), 255, dtype=np.uint8)
    strip = np.full((STRIP, width, 3), STATE_RGB[int(rendered.truth.state[t])], dtype=np.uint8)
    parts = [upscale(actual_image(rendered, t), scale), strip, gap, predicted_image(rendered, prediction, t, scale),
             gap, upscale(error_image(rendered, prediction, t, gain), scale)]
    return Image.fromarray(np.concatenate(parts, axis=0))


def frames_around_event(spec: SequenceSpec, before: int = 3, after: int = 4) -> list[int]:
    """Frames from a few before the digit touches the bar to a few after it leaves, or every frame from 1 without
    an event. Frame 0 is left out: its prediction is made before any input."""
    if spec.entry_frame < 0 or spec.exit_frame < 0:
        return list(range(1, spec.seq_len))
    return list(range(max(1, spec.entry_frame - before), min(spec.seq_len, spec.exit_frame + after + 1)))


def draw_row_names(draw: ImageDraw.ImageDraw, rendered: RenderedSequence, scale: int, top: int) -> None:
    """Write the row names in the left column, next to a tile whose top is at `top`."""
    height = rendered.spec.frame_height * scale
    for name, row_top in zip(ROW_NAMES, row_tops(rendered, scale)):
        draw.text((GAP, top + row_top + height // 2), name, fill=BLACK, font=font(10), anchor="lm")


def text_width(lines: tuple[str, ...], size: int) -> int:
    """Width in pixels of the longest line."""
    return int(max(font(size).getlength(line) for line in lines))


def prediction_sheet(rendered: RenderedSequence, prediction: np.ndarray, times: list[int], title: str,
                     scale: int = 2, columns: int = 10, gain: float = 2.0) -> Image.Image:
    """Tiles of the given frames in a grid, with the row names on the left, frame numbers, a title, and a legend."""
    tiles = [prediction_tile(rendered, prediction, t, scale, gain) for t in times]
    tile_w, tile_h = tiles[0].size
    columns = min(columns, len(tiles))
    rows = -(-len(tiles) // columns)
    lines = legend(gain)
    header = 3 * LINE + GAP
    width = max(LABEL_WIDTH + columns * (tile_w + GAP), text_width((title,), 12) + 2 * GAP,
                text_width(lines, 10) + 2 * GAP)
    sheet = Image.new("RGB", (width, header + rows * (tile_h + LINE + GAP) + GAP), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, GAP), title, fill=BLACK, font=font(12))
    for i, line in enumerate(lines):
        draw.text((GAP, GAP + (i + 1) * LINE), line, fill=(90, 90, 90), font=font(10))
    for i, (t, tile) in enumerate(zip(times, tiles)):
        left = LABEL_WIDTH + (i % columns) * (tile_w + GAP)
        top = header + (i // columns) * (tile_h + LINE + GAP)
        if i % columns == 0:
            draw_row_names(draw, rendered, scale, top)
        sheet.paste(tile, (left, top))
        draw.text((left + tile_w // 2, top + tile_h + 1), f"t={t}", fill=BLACK, font=font(10), anchor="ma")
    return sheet


def prediction_animation(renders: list[RenderedSequence], predictions: list[np.ndarray], labels: list[str],
                         times: list[int], scale: int = 2, gain: float = 2.0) -> list[Image.Image]:
    """One image per frame in `times`, with the sequences side by side, each under its label and its current state."""
    images = []
    for t in times:
        tiles = [prediction_tile(r, p, t, scale, gain) for r, p in zip(renders, predictions)]
        tile_w, tile_h = tiles[0].size
        top = 2 * LINE + GAP
        image = Image.new("RGB", (LABEL_WIDTH + len(tiles) * (tile_w + GAP), top + tile_h + GAP), WHITE)
        draw = ImageDraw.Draw(image)
        draw_row_names(draw, renders[0], scale, top)
        for i, (r, label, tile) in enumerate(zip(renders, labels, tiles)):
            left = LABEL_WIDTH + i * (tile_w + GAP)
            state = int(r.truth.state[t])
            draw.text((left, GAP), label, fill=BLACK, font=font(11))
            draw.text((left, GAP + LINE), f"t={t}  {STATE_NAMES[state]}", fill=STATE_RGB[state], font=font(11))
            image.paste(tile, (left, top))
        images.append(image)
    return images
