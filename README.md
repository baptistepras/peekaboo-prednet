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
|---|---|---|
| 1.1 | Project skeleton: device helper, seeding, configs, environment check, tests | done |
| 1.2 | MNIST digit pool: fixed splits, resized digits, ink sprites | in review |
| 1.3 to 1.11 | Data generator (trajectories, occluder, conditions, rendering, storage, validation, benchmarks) | next |
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
|---|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m scripts.env_check` | Lists the required packages and the selected device, then runs the PredNet operations forward and backward on that device. Setting the variable to 0 makes any operation that MPS does not support fail loudly instead of silently running on the CPU. | Every package `ok`, then `op check ok`. Exit code 0. |
| `python -m scripts.env_check --device cpu` | Same check, forced on the CPU (also accepts `cuda` or `mps`). | `op check ok`. |
| `python -m pytest` | Runs all unit tests in `tests/`. | All tests pass. |
| `python -m pytest tests/test_seeding.py -v` | Runs one test file with one line per test. | All tests pass. |
| `python -m scripts.prepare_mnist` | Downloads MNIST into `data/mnist/` (first run only), builds the train, val, and test digit pools, prints their statistics as JSON lines, checks that train and val share no digit, and saves a sheet of val sprites (one row per label) to `figures/mnist_pool.png`. Options: `--box-size`, `--ink-threshold`, `--val-size`, `--holdout-seed`, `--no-download`, `--sheet`. | Three JSON lines with 55000, 5000, and 10000 digits, then `train/val overlap: 0 digits, train + val = 60000`. |

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

### `scripts/prepare_mnist.py`

Downloads MNIST and reports on the three pools (see the Commands table). Look at `figures/mnist_pool.png` to check the resized digits.

### `scripts/env_check.py`

The environment report described in the Commands table.

## Conventions

- **Device agnostic.** The code runs on CUDA, MPS, or CPU. The device comes from `get_device`. Only float32 reaches the GPU, and CUDA only or MPS only operations are avoided.
- **Reproducible.** Fixed seeds, one config file per experiment, and every run saves a copy of its config next to its results in `runs/`.
- **Style.** American English, type annotations on every function, a short docstring per function.

## Repository layout

```
peekaboo/           the package
  device.py         device selection and helpers
  seeding.py        seeds
  config.py         YAML and JSON configs
  paths.py          standard locations
  data/             synthetic occlusion data
    mnist_pool.py   MNIST splits and digit sprites
scripts/            command line entry points
  env_check.py      environment report
  prepare_mnist.py  MNIST download and pool report
tests/              unit tests (pytest)
environment.yml     conda environment "peekaboo"
pyproject.toml      pytest settings (nothing to install)
THIRD_PARTY_NOTICES.md  references and licenses
```

These folders are created by the scripts and are not tracked by git: `data/`, `runs/`, `figures/`. The local references `papers/` and `third_party/` are not tracked either. The full list is in [`.gitignore`](.gitignore).

## References and licenses

This project reimplements PredNet (Lotter, Kreiman, and Cox, ICLR 2017) from the paper and uses the original code and OpenSTL as read only references. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
