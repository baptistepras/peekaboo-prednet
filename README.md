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
| 1.4 | Occluder bar: exact visible fractions and a placement solver for a target occlusion duration | done |
| 1.5 | Conditions (control, occlusion, hidden bounce, blackout) and PLATO style surprise tuples | done |
| 1.6 | Pixel rendering (observed, amodal, masks) and exact per frame ground truth | done |
| 1.7 | On the fly dataset, spawn safe loader, and stored datasets (write, read, verify) | done |
| 1.8 | Dataset validation: every sequence re-rendered and checked against the generator's rules | done |
| 1.9 | Contact sheets and GIFs of every condition and surprise | done |
| 1.10 | Generator benchmark, validation and test sets built from set configs | done |
| 1.11 | PredNet implementation and training benchmark on the target hardware | done |
| 2.1 | Checkpoints, and figures of the predictions next to the truth (hidden ink in cyan, error maps) | in review |
| 2 | Training loop, PredNet ablations, ConvLSTM, trackers, probes | planned |
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
| `python -m scripts.check_occluder` | For every cell (k, speed) of `configs/data/base.yaml`, places the occluder for 200 random val digits and prints one line per cell: share of digits placed, mean attempts, mean ink width of the placed digits (compare with the pool mean on the first line to spot a bias), bar width range (min, median, max), bar position range, onset frame range, and share of fully occluded frames. Saves space time diagrams to `figures/occluder_examples.png`. Needs MNIST (run `prepare_mnist` first). Options: `--config`, `--split`, `--n`, `--seed`, `--out`. | One line per cell, 100% placed except in the cells k=8 at v=4 and k=12 at v=3, where the widest digits (about 1 to 2%) do not fit. |
| `python -m scripts.check_conditions` | Generates 100 sequences of each condition, 1000 from the training mix, and 20 surprise tuples per surprise type and cell (k in {4, 8}, speed in {2, 4}, keeping only the speeds allowed for each surprise) on real val digits. Prints, per condition, the share built, the k values, the bar widths, the share of fully hidden frames, and the share of sequences with another full occlusion outside the analysis window. Prints the condition shares of the training mix and its share of hidden frames. Prints, per surprise type, the share of tuples built, whether every splice happens while hidden, and how much the reappearance moves. Saves `figures/conditions_examples.png` and `figures/surprise_tuples.png`. Needs MNIST. Options: `--config`, `--split`, `--n`, `--n-mix`, `--n-tuples`, `--seed`, `--out`. | Conditions built at or near 100%, training mix close to 30/60/10, about 20% hidden frames in occlusion sequences, `yes` in the splice column for every surprise. |
| `python -m scripts.preview_render` | Renders 300 sequences of the training mix, 50 blackouts, and 10 tuples of each surprise type on real val digits, and checks that the visible fraction, the true center, and the visible ink center measured on the pixels equal the exact ground truth. Also checks that the measured k equals the target and that no other contact with the bar falls inside an analysis window, and counts occlusion sequences with another full occlusion outside the window. Then saves `figures/render_preview.png`: for one sequence per condition and two surprises, a strip of observed frames (with the true center as a cross, cyan when hidden) and a strip of amodal frames. Needs MNIST. Options: `--config`, `--split`, `--n`, `--step`, `--seed`, `--out`. | Differences around 1e-15 or smaller, 0 k mismatches, 0 window intrusions, then `all checks passed`. |
| `python -m scripts.check_dataset` | Checks the data pipeline on real val digits. (1) Times an on the fly loader (20 batches of 16 sequences with 2 spawn workers by default) and prints the batch shapes. (2) Checks that the workers give exactly the batches of a single process. (3) Writes a small stored dataset (64 training mix sequences and one tuple of each surprise) to `data/datasets/smoke/`, reads it back, verifies every checksum, and compares its samples with the stream. Needs MNIST. Options: `--config`, `--split`, `--workers` (0 for a single process), `--batch-size`, `--batches`, `--seed`, `--out`. | A time per batch and a number of sequences per second, `yes` three times, then `all checks passed`. |
| `python -m scripts.validate_dataset --data data/datasets/smoke` | Validates a stored dataset: every sequence is re-rendered and checked (see `validate.py` for the list of checks). Prints one line per check with the number of sequences checked and failed, then statistics per condition. Saves the report as `validation.json` in the dataset folder. The generator settings come from the dataset's `info.yaml` (or `--config`). | Every check with 0 failures, then `all checks passed`. |
| `python -m scripts.bench_generator` | Times the generator on real val digits: milliseconds per sequence for each stage of a training sequence (spec, amodal frames, observed frames, ground truth), spec generation per condition, tuple generation per surprise type, then loader throughput with 0, 2, and 4 workers. Options: `--config`, `--split`, `--n`, `--tuples`, `--workers` (for example `0,2,4,8`), `--batch-size`, `--batches`, `--seed`. | A few milliseconds per sequence, and several hundred sequences per second with workers. |
| `python -m scripts.make_dataset --config configs/data/val_v1.yaml` | Builds the validation set (1000 sequences of the training mix, held out digits), writes it to `data/datasets/val_v1/`, then validates it and saves `validation.json` next to it. Options: `--overwrite` to rebuild, `--limit-per-cell N` for a quick trial with every cell capped at N (written as `<name>_limitN`), `--no-validate`. | A progress bar, the size on disk, then the validation report ending with `all checks passed`. |
| `python -m scripts.make_dataset --config configs/data/test_v1.yaml` | Builds the test set (22200 sequences, see "Validation and test sets" below) to `data/datasets/test_v1/` and validates it. | Same, for the test set. |
| `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m scripts.bench_prednet` | Trains PredNet (5 layers by default) for 10 warmup steps and 200 timed steps on the on the fly training stream, on the auto selected device. Prints the compute time and data wait per step, the sequences per second, the peak memory, the estimated time for 10k, 30k, and 50k steps, and the loss. Saves in `runs/bench/<model>_b<batch>_t<frames>_<device>/` a copy of the settings, `results.json`, `loss.png`, `predictions.png` (actual and predicted frames of 3 validation sequences, with the error of copying the last frame for reference; needs `val_v1`), and the trained model `model.pt` (a checkpoint for `show_predictions`). With the variable at 0, an operation MPS does not support fails instead of silently running on the CPU. Options: `--model` (for example `configs/models/prednet_4l.yaml`), `--steps`, `--warmup`, `--batch-size`, `--seq-len`, `--workers`, `--lr`, `--device`, `--seed`, `--out`. | Timings, a decreasing loss, and `saved in ...`. After 200 steps, PredNet is not expected to beat copying the last frame yet. |
| `python -m scripts.bench_prednet --device cpu --steps 5 --warmup 1 --batch-size 4 --seq-len 10 --workers 0` | CPU fallback check: the same code runs without a GPU. | A few steps and `saved in ...`. |
| `python -m scripts.show_predictions --run runs/bench/prednet_5l_b16_t40_mps` | Loads the checkpoint of a run (`model.pt`), predicts 4 occlusion sequences of `val_v1`, and saves in `<run>/predictions/` one sheet per sequence (`occlusion_<index>.png`, the frames around the crossing) and `occlusion.gif` (the 4 sequences side by side, every frame). Each frame shows three rows: the actual frame with the hidden ink in cyan, the prediction with the outline of the true digit (cyan where hidden), and the pixel error (red: in the frame but not predicted; blue: predicted but not in the frame). Prints the MSE of each sequence against copying the last frame. Nothing is trained. Options: `--checkpoint`, `--data` (set name or folder), `--condition` (or `any`), `--k`, `--speed`, `--n`, `--skip`, `--frames event` or `all`, `--step`, `--columns`, `--scale`, `--gain` (error brightness), `--fps`, `--device`, `--out`. | One MSE line per sequence, then `saved 5 files in ...`. |
| `python -m scripts.render_examples` | Draws one example of every condition and every surprise tuple on real val digits, and saves in `figures/generator/`: a contact sheet (`sheet_<condition>.png`, every other frame, observed above amodal) and a GIF (`<condition>.gif`) per condition; a contact sheet (`sheet_surprise_<kind>.png`, the four sequences A, B, AB, BA one under the other) and a GIF showing the four side by side (`surprise_<kind>.gif`) per surprise; and `overview.png` with all conditions. In every tile, the cross marks the true center (cyan when hidden) and the strip below gives the state. Needs MNIST. Options: `--config`, `--split`, `--index` (another example), `--seed`, `--scale`, `--step`, `--fps`, `--out`. | `saved 21 files in .../figures/generator` and their list. |
| `python -m scripts.validate_dataset --stream` | Same checks on sequences generated on the fly: 300 from the training mix and 5 tuples of each surprise type, plus a determinism check (the same index gives the same spec). Options: `--config`, `--split`, `--n`, `--tuples`, `--seed`. | Every check with 0 failures, then `all checks passed`. |

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
- `plan_crossing(rng, sprite, k, settings)`: the placement solver. It works event first. It draws a speed, a direction, an onset frame, and the position of the bar edge the digit enters through. It puts the digit just inside that edge at the onset frame and builds the trajectory outward from there. A bar of width ink width + (k - 1) x speed (plus a random sub step offset) hides the digit for exactly k frames geometrically. The solver also tries slightly narrower and wider bars on the exit side, because some digits have edge columns so faint (under 2% of the ink) that a nearly hidden frame already counts as occluded. It then **measures** k on the actual ink and keeps the placement only if all rules hold. Otherwise it draws again, so placements are uniform among the valid ones. The rules, over the analysis window (8 frames before entry to 4 frames after exit):
  - measured k equals the target;
  - the digit is fully visible for the 8 frames before entry and the 4 frames after exit;
  - no wall bounce on x from 2 frames before entry to 2 frames after exit, so the digit leaves on the far side, and no bounce on y in the 2 frames around entry and exit (a vertical bounce while hidden is allowed);
  - other contacts with the bar are allowed outside the window (`allow_contact_outside_window`).
- `CrossingSettings.from_config(config)`: the solver settings from `configs/data/base.yaml`.
- `plan_hidden_bounce(rng, sprite, k, settings)`: same idea with the bar against a wall. It anchors the trajectory just before the wall bounce and tries bar widths until the digit stays hidden for exactly k frames, with exactly one wall bounce around the event, while hidden.
- `plan_contact(rng, sprite, settings, speeds)`: a bar narrower than the digit, which passes behind it without ever being fully hidden anywhere in the sequence (the control).
- `windows_ok(...)` and `find_contact(...)`: the window rules and the contact episode, shared by the three solvers and by the surprise tuples.

**Why the frames are 96 px wide.** Along x, the digit needs room to approach the bar, to cross it, and to leave it. With ink width w, speed v, and occlusion duration k, the budget is about (k + 4.5) x v <= frame width - 3w. At 64 px, the widest digits could not be hidden for more than 1 frame at 4 px/frame, and the bar would sit almost always at the same place. At 96 px, every digit fits every cell of the grid except the widest ones (16 px, about 1% of digits) at k = 8 with v = 4 and at k = 12.

### `peekaboo/data/conditions.py`

Turns placements into complete sequence specs, one per condition:

| Condition | What happens | Used in |
|---|---|---|
| `control` | The bar is **narrower than the digit** (and at least 4 px wide). The digit passes behind it but is never fully hidden (k = 0). Everything else matches the occlusion sequences: a bar is present, the crossing has the same timing, and the same window rules apply. | train, test |
| `occlusion` | The digit crosses behind the bar and is fully hidden for exactly k frames. | train, test |
| `hidden_bounce` | The bar stands **against a wall**. The digit bounces off the wall while hidden and comes back out on the side it entered. Exactly one wall bounce happens around the event, while the digit is fully hidden. | train (10%), test |
| `blackout` | A control sequence whose frames are all set to zero for k frames, starting at the most covered frame. The bar stays present, as in training, and only the input disappears. | test |

Why the control uses a narrow bar: at 2 to 4 px/frame for 40 frames, the digit sweeps almost the whole frame width, and it cannot bounce off the bar. A bar that the digit never reaches would have to sit in a corner, far from where occlusion bars are, so it would not be a matched control.

- `GeneratorSettings.from_config(config)`: placement rules plus the training mix (30% control, 60% occlusion, 10% hidden bounce), the k weights, and the surprise parameters.
- `sample_spec(pool, settings, split, index, base_seed, condition=None, k=None, speed=None)`: builds sequence `index` of a split, fully determined by `(base_seed, split, index)`. Unset arguments are drawn: the condition from the training mix, k with weights proportional to k (so about 20% of the frames of occlusion sequences are fully hidden, as the plan targets), and the speed among those allowed for k. If a digit cannot be placed, another digit is drawn.
- `plan_condition`, `spec_from_plan`, `make_spec`: the steps behind `sample_spec`, reused by the surprise tuples.
- `spec_visible_fraction(spec, sprite)`: visible ink fraction per frame, 0 when the digit is absent or during a blackout.

Each spec records the key frames of its event (entry, onset, expected and actual reappearance, exit) and its **analysis window** (8 frames before entry to 4 frames after exit). Metrics are computed inside this window. The digit may meet the bar again outside it: about a third of the occlusion sequences contain a second full occlusion there. The ground truth labels those frames as "other" episodes (see `truth.py`), so metrics can leave them out.

### `peekaboo/data/splicing.py`

Surprise sequences, built as in the violation of expectation design of PLATO (Piloto et al., 2022). Each surprise is a **tuple of four sequences** with the same digit and bar:

- **possible A**: an ordinary occlusion;
- **possible B**: another plausible sequence, built from A's state at a splice frame where the digit is fully hidden, with one change applied;
- **impossible AB**: A's frames before the splice, then B's frames;
- **impossible BA**: B's frames before the splice, then A's frames.

A hidden digit is not drawn, so the frames at the splice are identical and the cut is invisible. The change happens strictly while the digit is hidden (it is hidden in both A and B at the frame before the splice and at the splice frame). Timing surprises (`speed_fast`, `speed_slow`, `early`) must also move the reappearance by at least one frame: a change made on the last hidden frame could otherwise leave it unchanged and go unnoticed. AB and B end with **identical frames** and differ only in their history, so the surprise signal is the model's error on AB after the splice minus its error on B, on the same frames.

| Surprise | Change applied at the splice | Effect on AB |
|---|---|---|
| `direction` | horizontal velocity reversed (the bar is not against a wall) | comes back out on the side it entered |
| `speed_fast` | speed doubled, from 2 to 4 px/frame | reappears early |
| `speed_slow` | speed halved, from 4 to 2 px/frame | reappears late |
| `offset` | vertical jump of 12 px | reappears at the right time, 12 px higher or lower |
| `early` | jump forward by half of the k hidden frames, same speed | reappears early |
| `vanish` | B is the same scene without any digit | never reappears (and BA appears from nowhere) |

B must move at a **training speed** (2, 3, or 4 px/frame). B is the matched control of the surprise: if it moved at 8 or 1 px/frame, speeds the models never saw, its own error would be high for a reason unrelated to surprise, and the difference between AB and B would no longer measure surprise. This is why the speed surprises are built from a single speed each. `surprise_speeds(kind, settings)` returns the allowed speeds, and other speeds are refused.

`direction` and `hidden_bounce` look alike at reappearance: the digit comes back out on the side it entered. Only the wall behind the bar makes one plausible and the other impossible, which makes this pair the cleanest test of RQ3.

- `sample_surprise_tuple(pool, settings, split, index, base_seed, kind, k, speed)`: one tuple, fully determined by its arguments. `tuple.specs()` returns A, B, AB, BA. Each spec carries `tuple_id`, `tuple_role`, and `surprise_frame` (the splice frame).

### `peekaboo/data/render.py`

Turns a spec into pixels. Frames are 64 x 96, stored as uint8.

- `render_amodal(spec, sprite)`: the **amodal** frames, one channel with the digit intensity, as if there were no bar and no blackout. Empty where the digit is absent (after a vanish).
- `compose_observed(amodal, spec, settings)`: the **observed** RGB frames that the models see. The digit is red (its intensity in the red channel, the other channels at 0), the background black, the bar a flat mid gray (128 on all channels) drawn on top, and blackout frames are all zero, bar included. The red and gray coding makes the digit and the bar trivial to tell apart, which the detector of the programmed baselines relies on.
- `modal_mask(amodal, spec)`: the ink pixels visible on screen.
- `render_sequence(spec, sprite, settings, thresholds)`: all of the above plus the ground truth, as a `RenderedSequence`.
- `measure_from_pixels(rendered)`: the visible fraction and both centers measured on the pixels, used to check the exact ground truth.

The observed frames and the masks are rebuilt exactly from the amodal frames and the spec, and the amodal frames from the spec and the sprite. Stored datasets therefore keep no frames at all (see `store.py`).

### `peekaboo/data/truth.py`

The ground truth of every frame, computed exactly from the spec and the sprite, without pixels, so it is cheap enough to compute during training. `compute_truth(spec, sprite, thresholds)` returns a `FrameTruth` with, per frame:

- `center`: the true (amodal) position of the digit, the intensity weighted centroid of all its ink, in pixels, even while hidden (NaN when the digit is absent);
- `modal_center`: the centroid of the visible ink only (NaN when nothing is visible), which is what a readout from the observed or predicted frames can recover during partial occlusion;
- `velocity`: the velocity leaving the frame, in px/frame;
- `visible_fraction`: the share of the ink visible on screen;
- `state`: `visible`, `partial`, `occluded`, `blackout`, or `absent`;
- `episode`: whether the frame is part of the **main** event (the contact with the bar that the sequence was built around), of **another** contact outside the analysis window, or of neither;
- `in_window`: whether the frame is inside the analysis window.

`sequence_summary(truth)` gives per sequence counts: the measured k of the main event, the number of other contacts and other full occlusions, and the number of other contact frames inside the window, which must be 0.

### `peekaboo/data/dataset.py`

PyTorch datasets. Every sample is a dictionary of tensors with the same keys, whatever its source:

| Key | Shape and type | Content |
| --- | --- | --- |
| `frames` | (T, 3, H, W) uint8 | observed frames, what the models see |
| `amodal` | (T, 1, H, W) uint8 | amodal frames, only if `include_amodal=True` |
| `center`, `modal_center` | (T, 2) float32 | true and visible ink centers, (y, x) in pixels, NaN when undefined |
| `velocity` | (T, 2) float32 | px/frame |
| `visible_fraction` | (T,) float32 | share of the ink visible on screen |
| `state`, `episode` | (T,) int64 | indices into `STATE_NAMES` and `EPISODE_*` of `truth.py` |
| `in_window` | (T,) bool | analysis window |
| `condition`, `tuple_role`, `surprise_type` | int64 | indices into `CONDITION_NAMES`, `ROLE_NAMES`, `SURPRISE_NAMES` |
| `index`, `k_target`, `speed`, event frames, `window_start`, `window_end`, `surprise_frame`, `tuple_id` | int64 | from the spec (-1 when unset) |

- `OnTheFlyDataset(pool, settings, render_settings, split, base_seed, ...)`: an endless stream for training. Sample `i` is always the same sequence, built from `(base_seed, split, i)`, whichever worker builds it, so no worker seeding is needed and a run can resume at any index. Condition, k, and speed can be fixed to restrict the stream to one cell.
- `IndexRangeSampler(start, count)` and `make_loader(dataset, batch_size, start, count, num_workers)`: a loader over indices `[start, start + count)`. Training step `s` with batch size `b` can start at `s * b`, so every sample is new and the order is reproducible. Workers always use spawn, the macOS default, so what works on the Mac also works on the cluster.
- `prepare_batch(batch, device)`: moves a batch to the device and turns the frames into float32 in [0, 1] there. Frames stay uint8 until then, which makes worker transfers four times smaller.

### `peekaboo/data/store.py`

Validation and test sets as small folders under `data/datasets/<name>/`:

| File | Content |
| --- | --- |
| `specs.jsonl` | one `SequenceSpec` per line, in dataset order |
| `sprites.npz` | the sprites of the digits used, so reading a set does not need MNIST |
| `truth.npz` | the exact ground truth of every frame, stacked over sequences |
| `metadata.parquet` | one row per sequence: spec fields, summary counts, and CRC32 checksums of its amodal and observed frames |
| `info.yaml` | format version, colors, thresholds, and how the set was built (generator config, seed) |

**No frames are stored.** Rendering is exact and costs about a millisecond per sequence, and storing frames would take about 2.5 GB for 10,000 test sequences of 64 x 96, against a few MB this way. The checksums prove that the frames rendered today are the frames rendered when the set was written, even if the code changes later. This refines decision D5, which planned to store the amodal frames.

- `write_dataset(directory, specs, pool, render_settings, thresholds, info, overwrite=False)`: writes a set. It builds the folder next to its destination and moves it in place only when complete, so an interrupted write never leaves a half written set.
- `StoredDataset(directory)`: reads a set and returns the same samples as `OnTheFlyDataset` for the same specs. `verify(i)` checks the checksums of sequence `i`, and `metadata` is the table as a pandas DataFrame.

### `peekaboo/data/validate.py`

Re-renders every sequence and checks it against the rules the generator promises. A `Validator` takes sequences one at a time, so any dataset size fits in memory, and checks each surprise tuple as soon as its four sequences have been seen.

| Check | Rule |
| --- | --- |
| `render_matches_truth` | visible fraction, true center, and visible ink center measured on the pixels equal the exact truth (this includes "the true center matches the centroid of the amodal mask" from the plan) |
| `colors` | the digit is pure red, the bar has its gray level, blackout frames are all zero |
| `window_clean` | fully visible context and post frames, and no other contact with the bar inside the analysis window |
| `exact_k` | the main event has exactly k consecutive occluded frames, starting at the onset frame |
| `clean_motion` | no wall bounce in the 2 frames around entry and exit, and exactly one bounce, while hidden, for `hidden_bounce` |
| `splice_hidden` | surprises change the trajectory strictly while the digit is hidden |
| `splice_identical` | AB and BA are pixel exact splices of A and B |
| `holdout` | validation sets use no digit of the training split |
| `stored_truth` | the stored ground truth equals the truth recomputed from the spec |
| `checksums` | frames render exactly as when the set was written |
| `determinism` | the same seed and index give the same sequence (stream only) |

The report also gives the number of sequences per condition and per cell, the share of occluded frames per condition (the plan targets 20 to 30% in occlusion sequences), and how many sequences have another full occlusion outside the window.

- `Validator(settings, render_settings, forbidden_digits)`, then `add(position, spec, sprite)` per sequence and `finish()` for the `ValidationReport`.
- `validate_stored(dataset, settings, forbidden_digits)`: the same on a stored set, including its stored truth and checksums.
- `training_digits(val_size, holdout_seed)`: the MNIST indices of the training split, forbidden in validation sets.
- `format_report(report)`: the report as text, as printed by the scripts.

### `peekaboo/data/build.py`

Builds the specs of a validation or test set from a **set config** (see "Validation and test sets" below). A set config lists plain sequences by condition, k, and speed, surprise tuples by kind, k, and speed, and optionally a number of sequences drawn from the training mix. Every combination of the listed values is one cell.

- `cells(set_config)` lists the cells and `count_sequences(set_config)` counts the sequences (a tuple counts as four).
- `build_specs(set_config, pool, settings, limit_per_cell=None)`: builds every spec. Each cell has its own random stream, derived from the set's base seed and the cell, so a set is rebuilt identically and cells do not depend on each other. Tuple ids are numbered across the whole set, and the four sequences of a tuple share one id.

### `peekaboo/models/prednet.py`

PredNet (Lotter, Kreiman, and Cox, ICLR 2017), reimplemented in PyTorch from the paper and checked against the original Keras code (read only reference, no code copied). Each layer l has a target A_l, a prediction Ahat_l, an error E_l, and a convolutional LSTM representation R_l.

- **Errors**: E_l = [ReLU(A_l - Ahat_l), ReLU(Ahat_l - A_l)], split into positive and negative populations.
- **Predictions**: Ahat_l = ReLU(conv(R_l)). For l = 0, the prediction is also clipped at the maximum pixel value (SatLU).
- **Targets**: A_0 is the frame, and A_{l+1} = maxpool(ReLU(conv(E_l))).
- **Representations**: R_l is a convolutional LSTM updated from E_l and R_l at the previous step and from the upsampled R_{l+1} at the current step. Its gates use the Keras 2 hard sigmoid, clip(0.2 x + 0.5, 0, 1), with no peepholes. PyTorch's own `hardsigmoid` has a different slope.
- **Update order**: each step first updates R from the top layer down, then computes Ahat, E, and A from the bottom up.
- **Loss**: the weighted mean activity of the error units. With "L0" the pixel layer has weight 1 and the others 0. The first step has weight 0 and the others 1 / (T - 1).
- **Initialization**: Glorot uniform kernels and zero biases, as in Keras.

`model(frames)` takes frames (B, T, 3, H, W) in [0, 1] and returns `prediction` (B, T, 3, H, W), where `prediction[:, t]` predicts frame t from the frames before it, and `layer_errors` (B, T, L). `return_states=True` also returns R and E of every layer at every step, for the probes and the error maps. `extrapolate_from=s` switches to closed loop from step s: the previous prediction replaces the input, as in the original code. `model.loss(layer_errors)` gives the training loss.

The configs `configs/models/prednet_5l.yaml` (channels 3, 16, 32, 64, 128) and `prednet_4l.yaml` (3, 32, 64, 128) both have about 3.1 million parameters. The 5 layer model reaches a 4 x 6 top layer on 64 x 96 frames, with a receptive field of 78 px per step, against 38 px for the 4 layer model (decision D9).

`peekaboo.models.build_model(config)` builds a model from a model config, chosen by its `model` key (`prednet` for now).

### `peekaboo/train/checkpoint.py`

- `save_checkpoint(path, model, optimizer, step, config, extra)`: saves the model and optimizer weights, the step, the run settings, and the Python, numpy, and torch random states (CPU, and CUDA or MPS when present). The file is written under a temporary name and then renamed, so an interrupted save never leaves a broken checkpoint.
- `load_checkpoint(path, model, optimizer, restore_rng)`: loads the weights into the model and optimizer when given, optionally restores the random states, and returns the whole checkpoint. The file holds only tensors and plain values and loads with `weights_only=True`, so loading never runs code. Tensors are read on the CPU and the model keeps its own device.

### `peekaboo/viz/space_time.py`

Space time diagrams shared by the report scripts: x horizontally, frames downward, the bar as a gray band, and the ink of each frame colored by visibility (green visible, orange partial, red fully hidden, black blackout, nothing when the digit is absent). `draw_spec(ax, spec, sprite, thresholds, title)` also marks the event frames, including the splice.

### `peekaboo/viz/frames.py`

Contact sheets and GIFs, drawn with Pillow so every pixel stays sharp. A frame tile shows the observed frame (what the models see), optionally the amodal frame below it, the true center of the digit as a cross (white when some ink is visible, cyan when hidden), and a strip colored by state (green visible, orange partial, red occluded, gray blackout, dark absent).

- `frame_tile(rendered, t, scale, show_amodal)`: one frame.
- `contact_sheet(rendered, title, step, scale, columns, show_amodal)`: every `step`-th frame in a grid, with frame numbers and a legend.
- `animation_frames(renders, labels, scale, show_amodal)` and `save_gif(images, path, fps)`: one image per time step with several sequences side by side (for example the four sequences of a surprise tuple), saved as a looping GIF.
- `stack(images)`: stacks sheets vertically.
- `describe(spec)`: a one line description of a sequence and its event frames, used in titles.

### `peekaboo/viz/predictions.py`

Figures of a model's predictions next to the truth. The prediction of a frame can never show a hidden digit: behind the bar, the correct prediction is the gray bar. These figures show what the model sees and predicts; what it believes about the hidden digit is read from its internal states by the probes of later steps. Each frame t is a tile of three rows:

| Row | Content |
| --- | --- |
| actual | the observed frame t, with every hidden ink pixel (under the bar or blacked out) blended toward cyan, and a strip colored by state below it |
| predicted | the model's prediction of frame t from frames 0 to t - 1, with a one pixel outline just outside the true digit: cyan next to hidden ink, white next to visible ink |
| error | PredNet's pixel error units E_0, brightened by a gain: red where the frame is brighter than the prediction (something missed), blue where the prediction is brighter (something predicted that is not there) |

The colors follow one convention in every figure of the project: red is ink on screen, cyan is the hidden truth, and later yellow will be the position a probe reads from the model and magenta the digit a decoder reads from it.

- `actual_image`, `predicted_image`, `error_image`: the three rows of one frame, as pixel arrays.
- `prediction_tile(rendered, prediction, t, scale, gain)`: the three rows stacked.
- `prediction_sheet(rendered, prediction, times, title, scale, columns, gain)`: the tiles of the chosen frames in a grid, with row names, frame numbers, a title, and a legend.
- `prediction_animation(renders, predictions, labels, times, scale, gain)`: one image per frame with several sequences side by side, for `save_gif`.
- `frames_around_event(spec)`: the frames from 3 before the digit touches the bar to 4 after it leaves (frame 0 is left out, since its prediction comes before any input).

### `peekaboo/data/spec.py`

`SequenceSpec` is the compact, complete description of one sequence: identity and seed, digit, per frame positions and velocities, whether the digit is present (it disappears after a "vanish" surprise), occluder bar, blackout frames, condition, k and speed, event frames (entry, onset, expected and actual reappearance, exit), analysis window, and for surprises the splice frame, tuple id, and role (A, B, AB, BA). Rendering a spec is deterministic, so test sets can be stored as specs. `metadata()` gives a flat row for a metadata table, and `to_dict()` and `from_dict()` convert to and from JSON.

### `scripts/prepare_mnist.py`

Downloads MNIST and reports on the three pools (see the Commands table). Look at `figures/mnist_pool.png` to check the resized digits.

### `scripts/plot_trajectories.py`

Visual check of the motion model (see the Commands table).

### `scripts/check_occluder.py`

Report on the placement solver with real digits, and space time diagrams: x horizontally, time downward, the bar as a gray band, and the ink extent of each frame in green (visible), orange (partial), or red (occluded), with the entry, onset, reappear, and exit frames marked.

### `scripts/check_conditions.py`

Report on conditions, training mix, and surprise tuples with real digits (see the Commands table), with space time diagrams of two sequences per condition and one tuple per surprise type.

### `scripts/preview_render.py`

Consistency check between the pixels and the exact ground truth on real digits, and a preview of the rendered frames (see the Commands table).

### `scripts/bench_prednet.py`

Training benchmark of PredNet on the target device, with a short run whose loss curve, predictions, and model are saved (see the Commands table).

### `scripts/show_predictions.py`

Prediction figures of a saved model on a stored set, drawn with `peekaboo/viz/predictions.py` (see the Commands table).

### `scripts/bench_generator.py`

Generator timing, stage by stage and per condition, and loader throughput (see the Commands table).

### `scripts/make_dataset.py`

Builds a set from its set config, writes it with `write_dataset`, and validates it (see the Commands table).

### `scripts/render_examples.py`

Contact sheets and GIFs of one example per condition and per surprise (see the Commands table).

### `scripts/validate_dataset.py`

Runs the validation on a stored dataset or on the stream, prints the report, and saves it next to a stored dataset (see the Commands table).

### `scripts/check_dataset.py`

Speed and determinism of the on the fly loader, and a full write, read, and verify cycle of a stored dataset (see the Commands table).

### `scripts/env_check.py`

The environment report described in the Commands table.

## Configs

- `configs/data/base.yaml`: generator settings shared by all splits. Frames of 64 x 96 pixels (height x width), sequences of 40 frames, digits resized to 20x20 with a 10% ink threshold, integer speeds |vx| in {2, 3, 4} and vy in {-2, ..., 2}, 8 fully visible frames before the digit reaches the bar, 4 after it leaves, and no wall bounce in the 2 frames around entry and exit. Visibility thresholds: occluded at most 2% visible ink, visible at least 95%. Occlusion durations k in {2, 4, 6, 8} at speeds {2, 3, 4}, and k = 12 at speeds {2, 3}. Control bars at least 4 px wide. Training mix 30% control, 60% occlusion, 10% hidden bounce, with k drawn with weights proportional to k. Surprises: 12 px vertical offset, a forward jump of half the hidden frames for "early", speed surprises from 2 to 4 and from 4 to 2 px/frame, and 100 placements of A tried per digit before another digit is drawn. Rendering: red digit, mid gray bar (0.5).
- `configs/data/val_v1.yaml` and `configs/data/test_v1.yaml`: the validation and test sets (next section).

## Validation and test sets

Training uses sequences generated on the fly from the training digits. Validation and test sets are fixed and stored, each built by `make_dataset` from a set config with its own base seed.

**`val_v1`** (1000 sequences): the training mix, with the 5000 held out digits of the MNIST train file. It is used for model selection and early stopping on the next frame loss only, never on occlusion metrics, so the research questions are not tuned on.

**`test_v1`** (22200 sequences), with digits of the MNIST test file (shapes never seen in training):

| Cells | k | Speed (px/frame) | Per cell | Sequences | Purpose |
| --- | --- | --- | --- | --- | --- |
| occlusion | 2, 4, 6, 8 | 2, 3, 4 | 400 | 4800 | RQ1 and RQ2: tracking through occlusion and recovery |
| occlusion | 12 | 2, 3 | 400 | 800 | longest occlusion (does not fit at 4 px/frame) |
| control | 0 | 2, 3, 4 | 400 | 1200 | same crossing without full occlusion |
| hidden_bounce | 4, 8 | 2, 3, 4 | 400 | 2400 | plausible reversal, the counterpart of `direction` |
| blackout | 2, 4, 8 | 2, 3, 4 | 200 | 1800 | missing input instead of an occluder |
| tuples: direction, offset, early, vanish | 4, 8 | 2, 3, 4 | 100 tuples | 9600 | RQ3: surprise |
| tuples: speed_fast | 4, 8 | 2 | 100 tuples | 800 | RQ3, B at 4 px/frame |
| tuples: speed_slow | 4, 8 | 4 | 100 tuples | 800 | RQ3, B at 2 px/frame |

400 sequences per cell give a standard error of about 0.25 px on a mean position error with a spread of 5 px. With 100 tuples per cell, the surprise AUROC has a standard error of about 0.04. On disk, the test set takes about 80 MB, since no frames are stored.

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
    occluder.py     visible fractions and the placement solvers
    conditions.py   control, occlusion, hidden bounce, blackout
    splicing.py     PLATO style surprise tuples
    render.py       observed and amodal frames, masks
    truth.py        exact per frame ground truth
    dataset.py      on the fly dataset, sampler, loader
    store.py        stored datasets: write, read, verify
    validate.py     dataset validation checks
    build.py        validation and test sets from set configs
  models/           video prediction models (build_model in __init__.py)
    prednet.py      PredNet
  train/            training
    checkpoint.py   save and load checkpoints
  viz/              figures
    space_time.py   space time diagrams
    frames.py       contact sheets and GIFs
    predictions.py  predictions next to the truth
scripts/            command line entry points
  env_check.py      environment report
  prepare_mnist.py  MNIST download and pool report
  plot_trajectories.py  visual check of the motion model
  check_occluder.py     placement report and space time diagrams
  check_conditions.py   conditions and surprise tuples report
  preview_render.py     pixels vs ground truth check and preview
  check_dataset.py      loader and stored dataset check
  validate_dataset.py   validation report of a dataset
  render_examples.py    contact sheets and GIFs of every condition
  bench_generator.py    generator and loader timing
  make_dataset.py       build, write, and validate a val or test set
  bench_prednet.py      PredNet training benchmark
  show_predictions.py   prediction figures of a saved model
configs/            one config file per experiment
  data/base.yaml    generator settings
  data/val_v1.yaml  validation set
  data/test_v1.yaml test set
  models/prednet_5l.yaml  PredNet, 5 layers
  models/prednet_4l.yaml  PredNet, 4 layers
tests/              unit tests (pytest)
environment.yml     conda environment "peekaboo"
pyproject.toml      pytest settings (nothing to install)
THIRD_PARTY_NOTICES.md  references and licenses
```

These folders are created by the scripts and are not tracked by git: `data/`, `runs/`, `figures/`. The local references `papers/` and `third_party/` are not tracked either. The full list is in [`.gitignore`](.gitignore).

## References and licenses

This project reimplements PredNet (Lotter, Kreiman, and Cox, ICLR 2017) from the paper and uses the original code and OpenSTL as read only references. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
