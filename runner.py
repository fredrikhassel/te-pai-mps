#!/usr/bin/env python
"""runner.py — Run all active experiments defined in config.json.

Reads config.json, merges each experiment's overrides onto the defaults,
and calls run_experiment.py as a subprocess for each active entry.

Usage:
    python runner.py                  # uses ./config.json
    python runner.py path/to/cfg.json # explicit config path
"""

import json
import subprocess
import sys
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_EXPERIMENT = os.path.join(_ROOT, "run_experiment.py")
RUN_SAMPLE_TRACKING = os.path.join(_ROOT, "run_sample_tracking.py")
RUN_RESOURCE_ESTIMATION = os.path.join(_ROOT, "run_resource_estimation.py")
RUN_TRUNCATION_COMPARISON = os.path.join(_ROOT, "run_truncation_comparison.py")
RUN_BOND_DIMENSION = os.path.join(_ROOT, "run_bond_dimension.py")
RUN_ERROR_TESTING = os.path.join(_ROOT, "run_error_testing.py")
RUN_TRUNCATION_COMPARISON_MULTI = os.path.join(_ROOT, "run_truncation_comparison_multi.py")
RUN_TROTTER_ACCURACY = os.path.join(_ROOT, "run_trotter_accuracy.py")
DATA_DIR = os.path.join(_ROOT, "data")

# Maps config key -> CLI flag for run_experiment.py
_FLAG_MAP = {
    "n_qubits":         "--n-qubits",
    "total_time":       "--total-time",
    "dt":               "--dt",
    "j":                "--j",
    "max_bond":         "--max-bond",
    "operator":         "--operator",
    "measure_qubit":    "--measure-qubit",
    "initial_state":    "--initial-state",
    "seed":             "--seed",
    "N_trotter":        "--N-trotter",
    "N_tepai":          "--N-tepai",
    "N_samples":        "--N-samples",
    "delta":            "--delta",
    "tepai_start_time": "--tepai-start-time",
    "n_cores":          "--n-cores",
    "N_trotter_canon":  "--N-trotter-canon",
    "trotter_levels":   "--trotter-levels",
    "plot_timeseries":  "--plot-timeseries",
    "max_bond_truncated": "--max-bond-truncated",
    "bd_option":          "--bd-option",
    "pi_over_deltas":     "--pi-over-deltas",
    "final_step_only":    "--final-step-only",
    "empirical_error":    "--empirical-error",
    "pi_over_delta":      "--pi-over-delta",
    "include_two_qubit":  "--include-two-qubit",
    "max_locality":       "--max-locality",
    "exact_threshold":    "--exact-threshold",
    "highlight_observable": "--highlight-observable",
    "N_max":                "--N-max",
    "N_canon":              "--N-canon",
}


def load_config(path):
    with open(path) as f:
        return json.load(f)


def build_cmd(params, plot_name, script=None):
    """Turn a flat param dict into a CLI arg list for an experiment script."""
    if script is None:
        script = RUN_EXPERIMENT
    cmd = [sys.executable, script, "--plot-name", plot_name]
    for key, flag in _FLAG_MAP.items():
        val = params.get(key)
        if val is not None:
            if isinstance(val, bool):
                if val:
                    cmd += [flag]
            elif isinstance(val, list):
                cmd += [flag] + [str(v) for v in val]
            else:
                cmd += [flag, str(val)]
    return cmd


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "config.json")
    cfg = load_config(cfg_path)

    defaults = dict(cfg.get("defaults", {}))

    # Resolve pi_over_delta -> delta in defaults (delta = pi / value)
    if "pi_over_delta" in defaults and "delta" not in defaults:
        import math
        defaults["delta"] = math.pi / defaults.pop("pi_over_delta")
    else:
        defaults.pop("pi_over_delta", None)

    experiments = cfg.get("experiments", [])
    active = [e for e in experiments if e.get("active", False)]

    if not active:
        print("No active experiments in config.")
        return

    print(f"Config: {cfg_path}")
    print(f"  {len(active)} active / {len(experiments)} total experiments\n")

    # Resolve params and build commands for every active experiment
    jobs = []  # list of (name, params, cmd)
    for i, exp in enumerate(active, 1):
        name = exp.get("name", f"experiment-{i}")

        # Merge: defaults <- experiment overrides
        exp_type = exp.get("type", "default")
        params = dict(defaults)
        for k, v in exp.items():
            if k in ("name", "active", "note", "type"):
                continue
            if k == "pi_over_delta":
                if exp_type == "resource_estimation":
                    # Pass through as list for --pi-over-delta
                    params["pi_over_delta"] = v if isinstance(v, list) else [v]
                    params.pop("delta", None)
                else:
                    import math
                    first = v[0] if isinstance(v, list) else v
                    params["delta"] = math.pi / first
            else:
                params[k] = v

        # Route to the appropriate script based on experiment type
        if exp_type == "sample_tracking":
            script = RUN_SAMPLE_TRACKING
        elif exp_type == "resource_estimation":
            script = RUN_RESOURCE_ESTIMATION
        elif exp_type == "truncation_comparison":
            script = RUN_TRUNCATION_COMPARISON
        elif exp_type == "truncation_comparison_multi":
            script = RUN_TRUNCATION_COMPARISON_MULTI
        elif exp_type == "bond_dimension":
            script = RUN_BOND_DIMENSION
        elif exp_type == "error_testing":
            script = RUN_ERROR_TESTING
        elif exp_type == "trotter_accuracy":
            script = RUN_TROTTER_ACCURACY
        else:
            script = RUN_EXPERIMENT

        plot_name = f"{name}.pdf"
        cmd = build_cmd(params, plot_name, script=script)
        jobs.append((name, params, cmd, exp.get("note")))

    # Split into single-sample (parallelisable) and multi-sample (sequential).
    # Experiments with N_samples=1 don't spawn internal workers so they are
    # safe to run concurrently across separate processes.
    single_sample = [(n, p, c, nt) for n, p, c, nt in jobs
                     if p.get("N_samples", 1) == 1]
    multi_sample  = [(n, p, c, nt) for n, p, c, nt in jobs
                     if p.get("N_samples", 1) != 1]

    failed = []
    finished = 0

    # --- parallel batch (N_samples == 1) ---
    if len(single_sample) > 1:
        n_workers = min(len(single_sample), mp.cpu_count() or 4)
        print(f"Launching {len(single_sample)} single-sample experiments "
              f"in parallel ({n_workers} workers)\n")
        for name, _, cmd, note in single_sample:
            print(f"  • {name}" + (f"  — {note}" if note else ""))
            print(f"    cmd: {' '.join(cmd)}")
        print()

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(subprocess.run, cmd): name
                for name, _, cmd, _ in single_sample
            }
            for fut in as_completed(futures):
                name = futures[fut]
                result = fut.result()
                elapsed = time.time() - t0
                if result.returncode != 0:
                    print(f"  ✗ {name} FAILED (exit {result.returncode}) "
                          f"[{elapsed:.1f}s elapsed]")
                    failed.append(name)
                else:
                    print(f"  ✓ {name} done [{elapsed:.1f}s elapsed]")
                finished += 1
        print()
    elif len(single_sample) == 1:
        # Only one single-sample experiment — just run it sequentially
        multi_sample = single_sample + multi_sample
        single_sample = []

    # --- sequential batch (N_samples > 1, or lone single-sample) ---
    for idx, (name, params, cmd, note) in enumerate(multi_sample, 1):
        seq_num = finished + idx
        print(f"{'=' * 60}")
        print(f"[{seq_num}/{len(jobs)}] {name}")
        if note:
            print(f"  {note}")
        print(f"{'=' * 60}")
        print(f"  cmd: {' '.join(cmd)}\n")

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  FAILED (exit {result.returncode}) after {elapsed:.1f}s\n")
            failed.append(name)
        else:
            print(f"\n  Completed in {elapsed:.1f}s\n")

    # Summary
    total = len(jobs)
    print(f"\n{'=' * 60}")
    print(f"  Finished: {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
