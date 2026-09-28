"""Save a contact sheet and a GIF for every condition and every surprise tuple, plus an overview sheet."""

import argparse
import sys
from pathlib import Path

from peekaboo.config import load_config
from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderedSequence, RenderSettings, render_sequence
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import SURPRISES, sample_surprise_tuple, surprise_speeds
from peekaboo.paths import CONFIGS_DIR, FIGURES_DIR
from peekaboo.viz.frames import animation_frames, contact_sheet, save_gif, stack

ROLES = ("possible A", "possible B", "impossible AB", "impossible BA")


def describe(spec: SequenceSpec) -> str:
    """A one line description of a sequence and its event frames."""
    parts = [f"{spec.condition}", f"digit {spec.label}", f"k={spec.k_target}", f"v={spec.speed}",
             f"bar {spec.bar_width} px"]
    for name, value in (("entry", spec.entry_frame), ("onset", spec.onset_frame),
                        ("reappears", spec.actual_reappear_frame), ("expected", spec.expected_reappear_frame),
                        ("splice", spec.surprise_frame)):
        if value >= 0 and not (name == "expected" and value == spec.actual_reappear_frame):
            parts.append(f"{name} t={value}")
    return ", ".join(parts)


def main() -> int:
    """Render one example of each condition and surprise and save the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0, help="which example to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=3, help="GIF pixel size (contact sheets use scale - 1)")
    parser.add_argument("--step", type=int, default=2, help="contact sheets show one frame out of this many")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "generator")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = GeneratorSettings.from_config(config)
    render_settings = RenderSettings.from_config(config)
    digits = config["digits"]
    pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    sheet_scale = max(1, args.scale - 1)

    def render(spec: SequenceSpec) -> RenderedSequence:
        """Render a spec with the configured colors."""
        return render_sequence(spec, pool.sprite(spec.digit_index), render_settings, settings.crossing.thresholds)

    args.out.mkdir(parents=True, exist_ok=True)
    saved, overview = [], []
    for condition in CONDITIONS:
        rendered = render(sample_spec(pool, settings, "examples", args.index, args.seed, condition=condition))
        title = describe(rendered.spec)
        path = args.out / f"sheet_{condition}.png"
        contact_sheet(rendered, title, step=args.step, scale=sheet_scale).save(path)
        saved.append(path)
        saved.append(save_gif(animation_frames([rendered], [title], scale=args.scale), args.out / f"{condition}.gif",
                              args.fps))
        overview.append(contact_sheet(rendered, title, step=args.step, scale=sheet_scale, show_amodal=False))

    for kind in SURPRISES:
        allowed = surprise_speeds(kind, settings)
        speed = 3 if 3 in allowed else allowed[0]
        renders = [render(s) for s in sample_surprise_tuple(pool, settings, "examples", args.index, args.seed,
                                                             kind, 6, speed).specs()]
        sheets = [contact_sheet(r, f"{kind}, {role}: {describe(r.spec)}", step=args.step, scale=sheet_scale,
                                show_amodal=False) for r, role in zip(renders, ROLES)]
        path = args.out / f"sheet_surprise_{kind}.png"
        stack(sheets).save(path)
        saved.append(path)
        labels = [f"{kind}: {role}" for role in ROLES]
        saved.append(save_gif(animation_frames(renders, labels, scale=args.scale, show_amodal=False),
                              args.out / f"surprise_{kind}.gif", args.fps))

    stack(overview).save(args.out / "overview.png")
    saved.append(args.out / "overview.png")
    print(f"saved {len(saved)} files in {args.out}:")
    for path in saved:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
