# MPS TE-PAI

**Code and data accompanying the paper on Time Evolution via Probabilistic Approximate Identity using Matrix Product States.**

This repository contains:

1. **`mps-te-pai`** -- a reusable Python package implementing the TE-PAI algorithm with MPS contraction.
2. **Experiment scripts** that reproduce the figures in the paper.
3. **Pre-computed datasets** so that figures can be regenerated from cached data without re-running the simulations.

This repository builds on the original TE-PAI codebase, adapting and extending it to the matrix-product-state setting used in this work. In particular, the present implementation restructures the core algorithm for MPS-based contraction, adds the experiment pipeline used for the results reported in the paper, and includes cached datasets for reproducibility. The original TE-PAI repository, on which this work is based, is linked here: https://github.com/CKiumi/te_pai.

---

## Table of Contents

- [MPS TE-PAI](#mps-te-pai)
  - [Table of Contents](#table-of-contents)
  - [The mps-te-pai Package](#the-mps-te-pai-package)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
    - [Package API](#package-api)
  - [Experiments](#experiments)
    - [Running Pre-configured Experiments](#running-pre-configured-experiments)
    - [Experiment Reference](#experiment-reference)
    - [Creating Your Own Experiment](#creating-your-own-experiment)
    - [Available Datasets](#available-datasets)
  - [compute the data from scratch, which requires significant computation time.](#compute-the-data-from-scratch-which-requires-significant-computation-time)
  - [Repository Structure](#repository-structure)
  - [Algorithm Overview](#algorithm-overview)
  - [Citation](#citation)
  - [License](#license)

---

## The mps-te-pai Package

The core simulation library lives in `mps-te-pai/` and is installable as a
standalone Python package. It provides:

- **Hamiltonian construction** for spin-chain Hamiltonians.
- **TE-PAI circuit generation** with configurable delta, Trotter step count, and circuit pool size.
- **MPS contraction** via [quimb](https://quimb.readthedocs.io/) with tuneable bond dimension.
- **Trotter reference simulation** for benchmarking.
- **Hybrid mode** where Trotter evolution covers early times and TE-PAI takes over from a specified start time.

### Installation

Requires Python >= 3.10.

```bash
# From the repository root:
pip install ./mps-te-pai

# Or in development mode:
pip install -e ./mps-te-pai
```

**Dependencies** (installed automatically): `numpy`, `scipy`, `quimb`, `numba`.

### Quick Start

```python
import numpy as np
from mps_tepai import simulate

result = simulate(
    n_qubits=10,
    T=1.0,              # total simulation time
    dt=0.1,             # snapshot interval
    delta=np.pi / 64,   # PAI rotation angle
    n_circuits=200,      # number of stochastic circuits
    N=100,             # Trotter steps in the TE-PAI decomposition per dT
    j=0.1,              # coupling strength
    max_bond=16,        # MPS bond dimension
    operator="X",
    measure_qubit=0,
    initial_state="plus_flip",
    seed=0,
)

# result.times           -- array of measurement times
# result.expectation_values -- TE-PAI estimates of <X_0>(t)
# result.std_errors      -- standard errors across circuits
```

For a more detailed example including Trotter comparison and plotting, see
`mps-te-pai/example.py`.

### Package API

**Top-level convenience function:**

| Function | Description |
|----------|-------------|
| `simulate(...)` | One-call interface: builds the Hamiltonian, generates circuits, runs MPS contraction, and returns a `SimulationResult`. |

**Core classes and functions (importable from `mps_tepai`):**

| Name | Description |
|------|-------------|
| `Hamiltonian` | Dataclass representing a Pauli Hamiltonian.|
| `TE_PAI` | Circuit generator. Takes a Hamiltonian and simulation parameters; call `run_te_pai(n_circuits)` to produce a list of `(signs, gates_arr)` circuit tuples. |
| `run_simulation(...)` | Contract a batch of TE-PAI circuits as MPS and return aggregated `SimulationResult`. |
| `run_simulation_hybrid(...)` | Hybrid mode: run Trotter up to a start time, then apply TE-PAI circuits from that point. |
| `trotter_mps(...)` | Pure Trotter reference simulation via MPS contraction. |
| `SimulationResult` | Dataclass with fields `times`, `expectation_values`, `std_errors`, `raw_runs`, `n_circuits`. |

**Submodules:**

| Module | Description |
|--------|-------------|
| `mps_tepai.hamiltonian` | `Hamiltonian` with L1-norm integration, and coefficient evaluation. |
| `mps_tepai.pai` | PAI decomposition: `abc(theta, delta)` coefficients, `prob_list(angles, delta)` probability vectors, `gamma(angles, delta)` normalisation factor. |
| `mps_tepai.sampling` | Numba-JIT parallel circuit sampling from probability arrays. |
| `mps_tepai.circuit_gen` | `TE_PAI` class for circuit generation. |
| `mps_tepai.mps_runner` | MPS circuit execution: `run_simulation`, `run_simulation_hybrid`, `trotter_mps`, gate application, and measurement. |

---

## Experiments

The root of the repository contains experiment runner scripts and a central
configuration file `config.json` that defines all experiments from the paper.

### Running Pre-configured Experiments

All experiments are defined in `config.json`. Each entry specifies the
experiment type, physical parameters, and algorithm settings. The `runner.py`
script reads this file and executes all experiments marked `"active": true`.

```bash
# Run all active experiments:
python runner.py

# Or run with a custom config:
python runner.py path/to/config.json
```

**To run a specific experiment**, set `"active": true` on the desired entry in
`config.json` (and `false` on the others), then run `python runner.py`.

Each experiment script automatically caches its computed data (CSV and NPZ files)
in a subfolder of `data/`. On subsequent runs with the same parameters, cached
data is loaded instead of recomputed. This means experiments with pre-computed
datasets (see [Available Datasets](#available-datasets)) complete in seconds.

**Output:** Each experiment produces a PDF plot saved in its data folder under
`data/`.

### Experiment Reference

| Config Name | Script | Figure | Description |
|-------------|--------|--------|-------------|
| `figure_2_q100` | `run_experiment.py` | Fig. 2 | 100-qubit TE-PAI with multiple start times (t=0, 1, 2) vs Trotter reference. |
| `fig_3` | `run_resource_estimation.py` | Fig. 3 | Resource estimation: TE-PAI vs Trotter gate counts for 50 qubits at varying Trotter depths. |
| `fig_4` | `run_sample_tracking.py` | Fig. 4 | Sample variance tracking over 1000 independent TE-PAI runs on 20 qubits. |
| `fig_5` | `run_error_testing.py` | Fig. 5 | Multi-observable error testing on 100 qubits across operators up to locality 3. |
| `figure_6A_q100_N20_d4096` | `run_experiment.py` | Fig. 6A | 100-qubit hybrid TE-PAI (Trotter until t=3, then TE-PAI). |
| `figure_6B_q100_N20_d4096` | `run_experiment.py` | Fig. 6B | 100-qubit full TE-PAI from t=0. |
| `fig_7` | `run_truncation_comparison.py` | Fig. 7 | 20-qubit MPS truncation comparison (chi=16 reference vs chi=2). |
| `fig_8` | `run_bond_dimension.py` | Fig. 8 | Bond-dimension study: TE-PAI starting at T/2. |
| `fig_9` | `run_bond_dimension.py` | Fig. 9 | Bond-dimension study: full run with multiple delta values. |

### Creating Your Own Experiment

There are two ways to set up a custom experiment:

**Option 1: Add an entry to `config.json`**

Add a new entry to the `"experiments"` list. Set `"active": true` and specify
the parameters and experiment type, then run `python runner.py`.

```json
{
  "name": "my_experiment",
  "active": true,
  "type": "default",
  "note": "My custom 20-qubit experiment",
  "n_qubits": 20,
  "total_time": 2.0,
  "N_trotter": 100,
  "N_tepai": 200,
  "pi_over_delta": 512,
  "N_samples": 10
}
```

Experiment types: `"default"` (standard TE-PAI vs Trotter), `"sample_tracking"`,
`"resource_estimation"`, `"truncation_comparison"`, `"bond_dimension"`,
`"error_testing"`, `"trotter_accuracy"`. Any parameter not specified inherits
from `"defaults"` in `config.json`.

**Option 2: Call a runner script directly**

Each runner script accepts CLI arguments:

```bash
python run_experiment.py \
  --n-qubits 20 \
  --total-time 2.0 \
  --dt 0.1 \
  --j 0.1 \
  --max-bond 16 \
  --N-trotter 100 \
  --N-tepai 200 \
  --N-samples 10 \
  --delta 0.006135923 \
  --plot-name my_plot.pdf
```

Run any script with `--help` to see all available options.

**Option 3: Use the `mps-te-pai` package directly**

For full programmatic control, import the package and build your simulation:

```python
import numpy as np
from mps_tepai import Hamiltonian, TE_PAI, run_simulation, trotter_mps

n_qubits = 20
T, dt, N = 2.0, 0.1, 200
delta = np.pi / 512
seed = 0

rng = np.random.default_rng(seed)
freqs = rng.uniform(-1, 1, size=n_qubits)
hamil = Hamiltonian.spin_chain(n_qubits, freqs, j=0.1)

n_snap = int(round(T / dt))

# Trotter reference
trotter_times, trotter_vals = trotter_mps(
    hamil, n_qubits, T=T, N=2000, n_snap=n_snap,
    max_bond=16, operator="X", measure_qubit=0,
    initial_state="plus_flip",
)

# TE-PAI
te_pai = TE_PAI(hamil, n_qubits, delta=delta, T=T, N=N, n_snap=n_snap)
circuits = te_pai.run_te_pai(n_circuits=200, seed=seed)
result = run_simulation(
    te_pai=te_pai, circuits=circuits,
    n_qubits=n_qubits, max_bond=16,
    operator="X", measure_qubit=0,
    initial_state="plus_flip",
)
```

### Available Datasets

The `data/` directory contains pre-computed results for the following experiments.
When you run these experiments, the runner scripts detect the cached files and
load them directly, skipping the computation.

| Experiment | Data Folder | Contents |
|------------|-------------|----------|
| Fig. 2 | `nq100_chi16_j0.10_seed0_T3.00_dt0.10_X0_plus_flip` | Trotter + TE-PAI at 3 start times |
| Fig. 3 | `nq50_chi8_j2.00_seed0_T2.50_dt0.10_X0_plus_flip` | Resource estimation Trotter sweeps + TE-PAI samples |
| Fig. 4 | `nq20_chi16_j0.10_seed0_T10.00_dt0.10_X0_plus_flip` | 1000-sample TE-PAI tracking + Trotter reference |
| Fig. 6A | `nq100_chi16_j0.10_seed0_T4.00_dt0.10_X0_plus_flip` | Hybrid Trotter+TE-PAI |
| Fig. 6B | `nq100_chi16_j0.10_seed0_T5.00_dt0.10_X0_plus_flip` | Full TE-PAI from t=0 |
| Fig. 8 & 9 | `nq10_chi16_j0.10_seed0_T1.00_dt0.10_X0_plus_flip` | Bond-dimension sweeps |

The data folder naming convention is
`nq{qubits}_chi{bond_dim}_j{coupling}_seed{seed}_T{time}_dt{timestep}_{operator}{qubit}_{initial_state}`.

**Note:** The dataset for the multi-observable error testing experiment (Fig. 5) and truncation experiment (Fig. 7) was too
large to store on GitHub and is not included. Running this experiment will
compute the data from scratch, which requires significant computation time.
---

## Repository Structure

```
TEPAI-MPS/
├── mps-te-pai/                    Reusable TE-PAI + MPS simulation package
│   ├── src/mps_tepai/             Package source
│   │   ├── __init__.py            Public API and simulate() convenience function
│   │   ├── hamiltonian.py         Hamiltonian definitions (spin chain, NNN, 2D)
│   │   ├── pai.py                 PAI decomposition coefficients and gamma factors
│   │   ├── sampling.py            Numba-JIT parallel circuit sampling
│   │   ├── circuit_gen.py         TE_PAI circuit generator class
│   │   └── mps_runner.py          MPS contraction, Trotter simulation, measurements
│   ├── tests/                     Test suite
│   ├── example.py                 Worked example comparing Trotter vs TE-PAI
│   └── pyproject.toml             Package metadata and dependencies
│
├── config.json                    Central experiment configuration
├── runner.py                      Orchestrator: runs all active experiments from config
├── run_experiment.py              Standard TE-PAI vs Trotter experiment
├── run_resource_estimation.py     Resource estimation (gate count comparison)
├── run_sample_tracking.py         Per-sample variance tracking
├── run_error_testing.py           Multi-observable error analysis
├── run_truncation_comparison.py   MPS truncation comparison (high vs low chi)
├── run_bond_dimension.py          Bond-dimension dependence study
├── run_trotter_accuracy.py        Trotter convergence accuracy
├── plot_style.py                  Shared matplotlib style configuration
│
├── data/                          Pre-computed experiment datasets
├── requirements.txt               Full Python dependencies
├── LICENSE                        MIT License
└── README.md
```

---

## Algorithm Overview

TE-PAI (Time Evolution via Probabilistic Approximate Identity) is a classical
simulation technique for quantum time evolution. Rather than tracking a full
quantum state, it generates a stochastic ensemble of quantum circuits whose
average expectation value converges to the true time-evolved observable.

Given a Hamiltonian H = sum_k h_k(t) P_k (a sum of Pauli terms) and a target
time T, TE-PAI:

1. **Trotterises** the evolution into N small time steps.
2. **Replaces each Pauli rotation** exp(-i theta P) with a probabilistic gate:
   with probabilities determined by the PAI decomposition at resolution delta,
   the rotation is replaced by the identity, a rotation by +/-delta, or a
   pi-rotation (which flips the circuit's sign contribution).
3. **Samples** many such stochastic circuits, each carrying a sign factor.
4. **Contracts** each circuit as an MPS using quimb and reads off expectation
   values at each time snapshot. The MPS bond dimension controls the accuracy
   of the contraction.
5. **Averages** the sign-weighted expectation values, scaled by the accumulated
   gamma factor, yielding a stochastic estimator of the target observable.

The key advantage is that each TE-PAI circuit contains far fewer gates than
the corresponding Trotter circuit (since most rotations are replaced by
the identity), at the cost of requiring multiple samples. The
parameter delta controls this trade-off: larger delta means fewer gates per
circuit but higher variance (larger gamma overhead).

---

## Citation

If you use this code in your research, please cite the accompanying paper.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
