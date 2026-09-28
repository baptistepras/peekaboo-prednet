"""Contact sheets and GIFs of rendered sequences, drawn with Pillow so every pixel stays sharp.

Each frame tile shows the observed frame (what the models see), optionally the amodal frame below it, the true
center of the digit as a cross (white when some ink is visible, cyan when hidden), and a strip colored by state.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from peekaboo.data.occluder import STATE_OCCLUDED
from peekaboo.data.render import RenderedSequence
from peekaboo.data.truth import STATE_BLACKOUT, STATE_NAMES

STATE_RGB = {0: (46, 160, 67), 1: (255, 140, 0), 2: (220, 50, 47), 3: (150, 150, 150), 4: (60, 60, 60)}
LEGEND = ("strip: green visible, orange partial, red occluded, gray blackout, dark absent; "
          "cross: true center (cyan when hidden)")
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GAP = 4    # pixels between tiles
STRIP = 5  # height of the state strip
LINE = 14  # height of one line of text


def font(size: int = 11) -> ImageFont.ImageFont:
    """Pillow's built in font at the given size (older Pillow versions ignore the size)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def upscale(image: np.ndarray, scale: int) -> np.ndarray:
    """Enlarge an image by an integer factor, with nearest neighbor pixels."""
    return image.repeat(scale, axis=0).repeat(scale, axis=1)


def frame_tile(rendered: RenderedSequence, t: int, scale: int = 3, show_amodal: bool = True) -> Image.Image:
    """Draw frame t: observed image, amodal image below it, true center cross, and state strip."""
    observed = upscale(rendered.observed[t], scale)
    width = observed.shape[1]
    parts = [observed]
    if show_amodal:
        amodal = np.repeat(rendered.amodal[t][..., None], 3, axis=2)
        parts += [np.full((GAP, width, 3), 255, dtype=np.uint8), upscale(amodal, scale)]
    state = int(rendered.truth.state[t])
    parts.append(np.full((STRIP, width, 3), STATE_RGB[state], dtype=np.uint8))
    tile = Image.fromarray(np.concatenate(parts, axis=0))

    cy, cx = rendered.truth.center[t]
    if not np.isnan(cy):
        hidden = state in (STATE_OCCLUDED, STATE_BLACKOUT)
        x, y, r = (cx + 0.5) * scale, (cy + 0.5) * scale, 2 * scale  # pixel i spans [i, i + 1) * scale
        draw = ImageDraw.Draw(tile)
        for start, end in (((x - r, y), (x + r, y)), ((x, y - r), (x, y + r))):
            draw.line([start, end], fill=(0, 255, 255) if hidden else WHITE, width=max(1, scale // 2))
    return tile


def contact_sheet(rendered: RenderedSequence, title: str, step: int = 2, scale: int = 2, columns: int = 10,
                  show_amodal: bool = True) -> Image.Image:
    """Lay out every `step`-th frame of a sequence in a grid, with its frame number and a title with the legend."""
    frames = list(range(0, rendered.spec.seq_len, step))
    tiles = [frame_tile(rendered, t, scale, show_amodal) for t in frames]
    tile_w, tile_h = tiles[0].size
    columns = min(columns, len(tiles))
    rows = -(-len(tiles) // columns)
    header = 2 * LINE + GAP
    sheet = Image.new("RGB", (GAP + columns * (tile_w + GAP), header + rows * (tile_h + LINE + GAP) + GAP), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, GAP), title, fill=BLACK, font=font(12))
    draw.text((GAP, GAP + LINE), LEGEND, fill=(90, 90, 90), font=font(10))
    for i, (t, tile) in enumerate(zip(frames, tiles)):
        left = GAP + (i % columns) * (tile_w + GAP)
        top = header + (i // columns) * (tile_h + LINE + GAP)
        sheet.paste(tile, (left, top))
        draw.text((left + tile_w // 2, top + tile_h + 1), f"t={t}", fill=BLACK, font=font(10), anchor="ma")
    return sheet


def stack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    """Stack images vertically, left aligned, on a white background."""
    width = max(image.width for image in images)
    out = Image.new("RGB", (width, sum(image.height for image in images) + gap * (len(images) - 1)), WHITE)
    top = 0
    for image in images:
        out.paste(image, (0, top))
        top += image.height + gap
    return out


def animation_frames(renders: list[RenderedSequence], labels: list[str], scale: int = 3,
                     show_amodal: bool = True) -> list[Image.Image]:
    """One image per time step, with the sequences side by side, each under its label and its current state."""
    seq_len = renders[0].spec.seq_len
    images = []
    for t in range(seq_len):
        tiles = [frame_tile(r, t, scale, show_amodal) for r in renders]
        tile_w, tile_h = tiles[0].size
        image = Image.new("RGB", (GAP + len(tiles) * (tile_w + GAP), 2 * LINE + GAP + tile_h + GAP), WHITE)
        draw = ImageDraw.Draw(image)
        for i, (r, label, tile) in enumerate(zip(renders, labels, tiles)):
            left = GAP + i * (tile_w + GAP)
            draw.text((left, GAP), label, fill=BLACK, font=font(11))
            state = STATE_NAMES[int(r.truth.state[t])]
            draw.text((left, GAP + LINE), f"t={t}  {state}", fill=STATE_RGB[int(r.truth.state[t])], font=font(11))
            image.paste(tile, (left, 2 * LINE + GAP))
        images.append(image)
    return images


def save_gif(images: list[Image.Image], path: str | Path, fps: float = 6.0) -> Path:
    """Save images as a looping GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000 / fps)), loop=0)
    return path
