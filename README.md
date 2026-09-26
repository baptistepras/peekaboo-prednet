# Peekaboo: Do Predictive Coding Networks Keep Track of Hidden Objects?

Controlled occlusion benchmark and analysis of PredNet: tracking through occlusion, reappearance, and surprise.

A digit moves behind an occluder and reappears, as expected or in a surprising way. We measure what a predictive coding video model (PredNet) predicts during and after the occlusion, what its internal state still encodes about the hidden object, and how strongly its prediction errors react to surprising reappearances. The project provides:

- a synthetic occlusion generator with full ground truth (hidden positions, amodal frames, scripted surprises);
- a clean PyTorch PredNet, with ablations, a ConvLSTM baseline, and programmed trackers (Kalman filters);
- a measurement protocol, latent probes, and visualizations.

## Research questions

1. **Tracking through occlusion.** While the object is hidden, how fast does each model lose it? We read the answer both from the predicted frames and, with linear probes, from the model's internal state.
2. **Correction after reappearance.** How many frames does each model need to recover when the object comes back, for expected and for surprising reappearances?
3. **Surprise.** Are prediction errors larger after a surprising reappearance than after a matched plausible one? Surprise tests follow the violation of expectation design of PLATO (Piloto et al., 2022): impossible and possible sequences end with the same frames and differ only in their history.

A constant velocity Kalman filter is close to optimal on this synthetic motion. It serves as a reference, not as a competitor.

## Status

| Phase | Content | State |
| --- | --- | --- |
| 1.1 | Project skeleton: device helper, seeding, configs, environment check, tests | done |
| 1.2 | MNIST digit pool: fixed splits, resized digits, ink sprites | done |
| 1.3 | Sequence spec and bouncing trajectories built from an anchor frame | done |
| 1.4 | Occluder bar: exact visible fractions and a placement solver for a target occlusion duration | in review |
| 1.5 to 1.11 | Data generator (conditions, rendering, storage, validation, benchmarks) | next |
| 2 | PredNet, ablations, ConvLSTM, trackers, training | planned |
| 3 | Evaluation, probes, figures | planned |

## Setup

The project has its own environment, `peekaboo`, defined in [`environment.yml`](environment.yml). All packages come from conda-forge. The same file works on macOS (Apple GPU through MPS) and on a Linux CUDA cluster.

```
mamba env create -f environment.yml          # create the environment once
mamba activate peekaboo                      # activate it in every new terminal
mamba env update -f environment.yml --prune  # after environment.yml changes
```

Every command below runs from the project root with `peekaboo` active. The package is not installed. Scripts run with `python -m scripts.<name>`, which puts the project root on the import path.

## Commands

| Command | What it does | Expected result |
| --- | --- | --- |
| `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m scripts.env_check` | Lists the required packages and the selected device, then runs the PredNet operations forward and backward on that device. Setting the variable to 0 makes any operation that MPS does not support fail loudly instead of silently running on the CPU. | Every package `ok`, then `op check ok`. Exit code 0. |
| `python -m scripts.env_check --device cpu` | Same check, forced on the CPU (also accepts `cuda` or `mps`). | `op check ok`. |
| `python -m pytest` | Runs all unit tests in `tests/`. | All tests pass. |
| `python -m pytest tests/test_seeding.py -v` | Runs one test file with one line per test. | All tests pass. |
| `python -m scripts.prepare_mnist` | Downloads MNIST into `data/mnist/` (first run only), builds the train, val, and test digit pools, prints their statistics as JSON lines, checks that train and val share no digit, and saves a sheet of val sprites (one row per label) to `figures/mnist_pool.png`. Options: `--box-size`, `--ink-threshold`, `--val-size`, `--holdout-seed`, `--no-download`, `--sheet`. | Three JSON lines with 55000, 5000, and 10000 digits, then `train/val overlap: 0 digits, train + val = 60000`. |
| `python -m scripts.plot_trajectories` | Samples 12 random trajectories with the motion settings of `configs/data/base.yaml` and saves their paths to `figures/trajectories.png`, with wall bounces (red crosses) and anchor frames (black stars). Options: `--config`, `--n`, `--seed`, `--out`. | `saved .../figures/trajectories.png`. Paths stay inside the frame and mirror off the borders. |
| `python -m scripts.check_occluder` | For every cell (k, speed) of `configs/data/base.yaml`, places the occluder for 200 random val digits and prints one line per cell: share of digits placed, mean attempts, mean ink width of the placed digits (compare with the pool mean on the first line to spot a bias), bar width range (min, median, max), bar position range, onset frame range, and share of fully occluded frames. Saves space time diagrams to `figures/occluder_examples.png`. Needs MNIST (run `prepare_mnist` first). Options: `--config`, `--split`, `--n`, `--seed`, `--out`. | One line per cell, placed close to 100% except possibly the cells k=8 at v=4 and k=12, where the widest digits do not fit. |

## Modules

### `peekaboo/device.py`

The only place where the compute device is chosen. No other code hardcodes a device.

- `get_device(preference="auto")`: returns CUDA if available, else MPS, else CPU. An explicit `"cuda"`, `"mps"`, or `"cpu"` is honored, and an unavailable backend raises an error instead of falling back silently.
- `describe_device(device)`: short readable name, used in logs.
- `synchronize(device)`: waits for queued GPU work, so timings are correct on CUDA and MPS.
- `to_tensor(array, device)`: converts a numpy array or tensor to float32 on the CPU, then moves it. This keeps float64 away from MPS, which does not support it.

### `peekaboo/seeding.py`

- `derive_seed(*keys)`: turns keys such as `(base_seed, "train", index)` into a stable 63 bit seed. It uses a cryptographic hash, so the result is the same in every process and on every machine. The generator uses it to build each sequence reproducibly, even with several data loading workers.
- `make_rng(*keys)`: a numpy random generator seeded from the same keys.
- `seed_everything(seed, deterministic=False)`: seeds Python, numpy, and torch. With `deterministic=True` it also requests deterministic kernels where they exist. Data generation is exactly reproducible. Training on a GPU can still differ slightly between runs.
- `worker_init_fn(worker_id)`: gives each data loading worker its own seed.

### `peekaboo/config.py`

Experiments are described by one config file each, in YAML or JSON.

- `load_config(path)` and `save_config(config, path)`: the format follows the file extension.
- `apply_overrides(config, ["train.lr=0.001", "data.seq_len=30"])`: changes values from the command line. Values are parsed as YAML, so numbers and lists get the right type. A misspelled key raises an error.

### `peekaboo/paths.py`

Standard locations, resolved from the package location: `data/`, `data/mnist/`, `data/datasets/`, `runs/`, `figures/`, `configs/`.

### `peekaboo/data/mnist_pool.py`

The digits used by the generator. Test sequences only use digits from the MNIST test file, and validation sequences only use a holdout of the MNIST train file, so both are unseen shapes for the trained models.

- Splits: `train` is the MNIST train file minus a fixed random holdout of 5000 digits, `val` is that holdout, `test` is the full MNIST test file (10000 digits). `split_indices(split, n_images, val_size, holdout_seed)` returns the indices, identical on every machine.
- `prepare_sprite(image, box_size=20, ink_threshold=0.1)`: resizes a 28x28 digit to 20x20 (the whole image, so the ink shrinks from about 20 to about 14 pixels), sets pixels below 10% intensity to zero, and crops to the ink. After this step every nonzero pixel is ink, which makes ink masks and visible fractions exact.
- `DigitPool`: all sprites of one split, padded into one array, with their MNIST index, label, ink height and width, total ink, and intensity weighted centroid. The centroid is the reference point for the ground truth position of the digit. `pool.sprite(i)` returns one tight sprite, `pool.sample(rng)` draws one digit.
- `build_digit_pool(split)`: loads MNIST through torchvision and builds the pool of a split. `pool_summary(pool)` gives label counts and ink size percentiles.

### `peekaboo/data/trajectory.py`

Integer motion of the digit, with bounces off the frame borders. Positions are the top left corner of the tight ink sprite, so the ink itself touches the border when it bounces.

- Trajectories are built **outward from an anchor frame**: the state (position and velocity) is fixed at one frame, for example the first fully hidden frame, and the motion is extended forward and backward in time. This lets the generator place the occlusion event first and derive the rest of the sequence from it.
- Each axis is uniform motion on an unfolded line, folded back into the allowed range. Bounces are therefore exactly reversible: rebuilding a trajectory from its state at any other frame gives the same trajectory. A sprite resting on a wall always has its velocity pointing inward.
- `Bounds.for_sprite(frame_height, frame_width, height, width)`: allowed range of the top left corner for a sprite of that ink size.
- `build_trajectory(anchor_frame, anchor_pos, anchor_vel, seq_len, bounds)`: returns a `Trajectory` with per frame `positions`, `velocities` (velocity leaving each frame), and `bounced` (a wall bounce between the previous frame and this one), all in (y, x) order.
- `continue_from(trajectory, frame, pos, vel, bounds)`: keeps the past and restarts from a new state at `frame`. Surprise events (direction reversal, speed change, teleport) will use it.
- `has_bounce(trajectory, first, last)`: checks the no bounce windows around the occlusion.
- `sample_velocity(rng, x_speeds, y_velocities)`: draws an integer velocity, with a random sign for vx.

### `peekaboo/data/occluder.py`

The occluder is a full height vertical gray bar, drawn on top of the digit. This module measures what the bar hides and places it so that the digit stays fully hidden for exactly k frames.

- `visible_fraction(column_ink(sprite), x_positions, bar_left, bar_width)`: fraction of the digit's ink that the bar leaves visible, per frame. Since the bar covers whole columns, this is exact and cheap. It is computed on the ink, not on the digit's bounding box, as the plan requires.
- `visibility_states(fractions, thresholds)`: each frame is *occluded* (at most 2% of the ink visible), *visible* (at least 95%), or *partial*.
- `find_crossing(fractions, thresholds, frame)`: the occlusion episode that contains `frame`, as a `CrossingEvent` with four key frames: **entry** (the bar first covers ink), **onset** (first fully occluded frame), **reappear** (first frame with visible ink again), and **exit** (last frame with covered ink). `event.k = reappear - onset`.
- `plan_crossing(rng, sprite, k, settings)`: the placement solver. It works event first. It draws a speed, a direction, and an onset frame. It sizes the bar so the digit is hidden for exactly k frames (width = ink width + (k - 1) x speed + a random sub step offset), places the bar at random, puts the digit at the bar edge at the onset frame, and builds the trajectory outward from there. It then **measures** k on the actual ink and keeps the placement only if all rules hold. Otherwise it tries again, so placements are uniform among the valid ones. The rules, over the analysis window (8 frames before entry to 4 frames after exit):
  - measured k equals the target;
  - the digit is fully visible for the 8 frames before entry and the 4 frames after exit;
  - no wall bounce on x from 2 frames before entry to 2 frames after exit, so the digit leaves on the far side, and no bounce on y in the 2 frames around entry and exit (a vertical bounce while hidden is allowed);
  - other contacts with the bar are allowed outside the window (`allow_contact_outside_window`).
- `CrossingSettings.from_config(config)`: the solver settings from `configs/data/base.yaml`.

**Why the frames are 96 px wide.** Along x, the digit needs room to approach the bar, to cross it, and to leave it. With ink width w, speed v, and occlusion duration k, the budget is about (k + 4.5) x v <= frame width - 3w. At 64 px, the widest digits could not be hidden for more than 1 frame at 4 px/frame, and the bar would sit almost always at the same place. At 96 px, every digit fits every cell of the grid except the widest ones (16 px, about 1% of digits) at k = 8 with v = 4 and at k = 12.

### `peekaboo/data/spec.py`

`SequenceSpec` is the compact, complete description of one sequence: identity and seed, digit, per frame positions and velocities, whether the digit is present (it disappears after a "vanish" surprise), occluder bar, blackout frames, condition, and event frames (occlusion onset, expected and actual reappearance, surprise). Rendering a spec is deterministic, so test sets can be stored as specs. `metadata()` gives a flat row for a metadata table, and `to_dict()` and `from_dict()` convert to and from JSON.

### `scripts/prepare_mnist.py`

Downloads MNIST and reports on the three pools (see the Commands table). Look at `figures/mnist_pool.png` to check the resized digits.

### `scripts/plot_trajectories.py`

Visual check of the motion model (see the Commands table).

### `scripts/check_occluder.py`

Report on the placement solver with real digits, and space time diagrams: x horizontally, time downward, the bar as a gray band, and the ink extent of each frame in green (visible), orange (partial), or red (occluded), with the entry, onset, reappear, and exit frames marked.

### `scripts/env_check.py`

The environment report described in the Commands table.

## Configs

- `configs/data/base.yaml`: generator settings shared by all splits. Frames of 64 x 96 pixels (height x width), sequences of 40 frames, digits resized to 20x20 with a 10% ink threshold, integer speeds |vx| in {2, 3, 4} and vy in {-2, ..., 2}, 8 fully visible frames before the digit reaches the bar, 4 after it leaves, and no wall bounce in the 2 frames around entry and exit. Visibility thresholds: occluded at most 2% visible ink, visible at least 95%. Occlusion durations k in {2, 4, 6, 8} at speeds {2, 3, 4}, and k = 12 at speeds {2, 3}. Later tasks add the condition section.

## Conventions

- **Device agnostic.** The code runs on CUDA, MPS, or CPU. The device comes from `get_device`. Only float32 reaches the GPU, and CUDA only or MPS only operations are avoided.
- **Reproducible.** Fixed seeds, one config file per experiment, and every run saves a copy of its config next to its results in `runs/`.
- **Style.** Type annotations on every function, a short docstring per function.

## Repository layout

```
peekaboo/           the package
  device.py         device selection and helpers
  seeding.py        seeds
  config.py         YAML and JSON configs
  paths.py          standard locations
  data/             synthetic occlusion data
    mnist_pool.py   MNIST splits and digit sprites
    trajectory.py   bouncing integer trajectories from an anchor frame
    spec.py         SequenceSpec, the description of one sequence
    occluder.py     visible fractions and the occluder placement solver
scripts/            command line entry points
  env_check.py      environment report
  prepare_mnist.py  MNIST download and pool report
  plot_trajectories.py  visual check of the motion model
  check_occluder.py     placement report and space time diagrams
configs/            one config file per experiment
  data/base.yaml    generator settings
tests/              unit tests (pytest)
environment.yml     conda environment "peekaboo"
pyproject.toml      pytest settings (nothing to install)
THIRD_PARTY_NOTICES.md  references and licenses
```

These folders are created by the scripts and are not tracked by git: `data/`, `runs/`, `figures/`. The local references `papers/` and `third_party/` are not tracked either. The full list is in [`.gitignore`](.gitignore).

## References and licenses

This project reimplements PredNet (Lotter, Kreiman, and Cox, ICLR 2017) from the paper and uses the original code and OpenSTL as read only references. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
