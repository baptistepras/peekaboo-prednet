"""SequenceSpec: the complete, compact description of one sequence. Rendering a spec is fully deterministic."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

_ARRAY_FIELDS = ("positions", "velocities", "digit_present")


@dataclass(frozen=True)
class SequenceSpec:
    """Everything needed to render one sequence and to interpret its ground truth.

    Positions are the top left corner of the tight ink sprite, in (y, x) pixel order. Unset event frames are -1.
    """

    # identity
    split: str
    index: int
    seed: int
    # scene
    frame_height: int
    frame_width: int
    seq_len: int
    # digit (see peekaboo.data.mnist_pool.DigitPool)
    digit_index: int
    mnist_index: int
    label: int
    ink_height: int
    ink_width: int
    # motion, one row per frame
    positions: np.ndarray       # (seq_len, 2) int64
    velocities: np.ndarray      # (seq_len, 2) int64, velocity leaving each frame
    digit_present: np.ndarray   # (seq_len,) bool, False after a "vanish" event
    # occluder: a full height vertical bar drawn on top of the digit, width 0 means no bar
    bar_left: int = 0
    bar_width: int = 0
    blackout_frames: tuple[int, ...] = ()
    # condition and events
    condition: str = "control"
    k_target: int = 0
    speed: int = 0                     # |vx| at the event
    entry_frame: int = -1              # first frame where the bar covers some ink
    onset_frame: int = -1              # first fully hidden frame (most covered frame for the control)
    expected_reappear_frame: int = -1  # first frame with visible ink if nothing unexpected happens
    actual_reappear_frame: int = -1    # first frame with visible ink in this sequence
    exit_frame: int = -1               # last frame where the bar covers some ink
    window_start: int = -1             # analysis window: context frames before the event ...
    window_end: int = -1               # ... to post frames after it, both inclusive
    surprise_type: str = "none"
    surprise_frame: int = -1           # first frame of the changed trajectory (the splice frame)
    tuple_id: int = -1                 # links the four sequences of one surprise tuple
    tuple_role: str = "none"           # "A", "B" (possible), "AB", "BA" (impossible)
    extra: dict[str, Any] = field(default_factory=dict)  # condition specific parameters

    def metadata(self) -> dict[str, Any]:
        """Return the scalar fields as a flat dictionary, for one row of a metadata table."""
        row = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _ARRAY_FIELDS}
        row["blackout_frames"] = ",".join(str(t) for t in self.blackout_frames)
        extra = row.pop("extra")
        row.update({f"extra_{key}": value for key, value in extra.items()})
        return row

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly dictionary (arrays become nested lists)."""
        data = asdict(self)
        for name in _ARRAY_FIELDS:
            data[name] = getattr(self, name).tolist()
        data["blackout_frames"] = list(self.blackout_frames)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SequenceSpec":
        """Rebuild a spec from the output of to_dict."""
        data = dict(data)
        data["positions"] = np.asarray(data["positions"], dtype=np.int64).reshape(-1, 2)
        data["velocities"] = np.asarray(data["velocities"], dtype=np.int64).reshape(-1, 2)
        data["digit_present"] = np.asarray(data["digit_present"], dtype=bool)
        data["blackout_frames"] = tuple(int(t) for t in data.get("blackout_frames", ()))
        return cls(**data)

    def __eq__(self, other: object) -> bool:
        """Compare field by field, using array equality for the per frame arrays."""
        if not isinstance(other, SequenceSpec):
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if f.name in _ARRAY_FIELDS:
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True
