"""MPS execution engine for TE-PAI circuits."""

from dataclasses import dataclass
from typing import List, Optional
import warnings
import numpy as np

# Suppress cotengra "no optimizer found" warning before importing quimb
warnings.filterwarnings("ignore", message=".*optuna.*cmaes.*nevergrad.*", category=UserWarning)

import quimb.tensor as qtn
import quimb as qu
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class SimulationResult:
    """Results from a TE-PAI MPS simulation.

    Attributes:
        times: Time points [0, dT, 2*dT, ...].
        expectation_values: Mean expectation value at each time.
        std_errors: Standard error of the mean at each time.
        raw_runs: Per-circuit weighted results (list of lists).
        n_circuits: Number of circuits used.
        overhead: Sampling overhead factor (gamma).
    """
    times: np.ndarray
    expectation_values: np.ndarray
    std_errors: np.ndarray
    raw_runs: List[List[float]]
    n_circuits: int
    overhead: float


# Gate name mapping from Hamiltonian convention to quimb gate IDs
_GATE_MAP = {"XX": "rxx", "YY": "ryy", "ZZ": "rzz", "Z": "rz"}


def create_mps_circuit(n_qubits, initial_state="plus_flip", max_bond=None):
    """Create a quimb MPS circuit with the given initial state.

    Args:
        n_qubits: Number of qubits.
        initial_state: One of 'plus' (all |+>), 'zero' (all |0>),
                       or 'plus_flip' (all |+> with Z on middle qubit).
        max_bond: Maximum MPS bond dimension. None for no truncation.
    """
    if max_bond is None:
        circ = qtn.CircuitMPS(n_qubits, cutoff=1e-12)
    else:
        circ = qtn.CircuitMPS(n_qubits, max_bond=max_bond, cutoff=1e-12)

    if initial_state in ("plus", "plus_flip"):
        for i in range(n_qubits):
            circ.apply_gate("H", qubits=[i])
        if initial_state == "plus_flip":
            middle = int(np.floor(n_qubits / 2))
            circ.apply_gate("Z", qubits=[middle])
    elif initial_state == "zero":
        pass  # already |0>^n
    else:
        raise ValueError(f"Unknown initial_state: {initial_state}")

    return circ


def apply_gates(circuit, gates):
    """Apply a list of (pauli, angle, qubits) gates to an MPS circuit."""
    for gate in gates:
        gate_name, angle, qubit_indices = gate[0], gate[1], gate[2]
        quimb_name = _GATE_MAP[gate_name]
        if len(qubit_indices) == 1:
            circuit.apply_gate(gate_id=quimb_name, qubits=[qubit_indices[0]], params=[angle])
        elif len(qubit_indices) == 2:
            circuit.apply_gate(gate_id=quimb_name, qubits=qubit_indices, params=[angle])
        else:
            raise ValueError(f"Unsupported qubit count for gate {gate_name}: {len(qubit_indices)}")


def measure(circuit, operator="X", qubit=0):
    """Measure expectation value of a Pauli operator on a qubit.

    Args:
        circuit: quimb MPS circuit.
        operator: Pauli operator ('X', 'Y', or 'Z').
        qubit: Qubit index to measure.

    Returns:
        Expectation value mapped to [0, 1] range: (val + 1) / 2.
    """
    val = np.real(circuit.local_expectation(qu.pauli(operator), (qubit,)))
    return (val + 1) / 2


def _run_single(run_data):
    """Execute a single TE-PAI circuit as a full MPS run.

    The circuit's gates_arr contains n_snap sublists (one per snapshot).
    We apply each sublist sequentially, measuring after each.
    signs[k] gives the cumulative pi-parity through snapshot k.
    """
    n_qubits, signs, gates_arr, gam_list, max_bond, initial_state, operator, meas_qubit = run_data

    circ = create_mps_circuit(n_qubits, initial_state, max_bond)
    results = []

    # t=0 measurement (gamma_0 = 1)
    results.append(measure(circ, operator, meas_qubit))

    for snap_idx, snapshot_gates in enumerate(gates_arr):
        apply_gates(circ, snapshot_gates)
        measured = measure(circ, operator, meas_qubit)
        weight = signs[snap_idx] * gam_list[snap_idx + 1]
        results.append(measured * weight)

    return results


def _run_single_hybrid(run_data):
    """Execute a Trotter prefix followed by TE-PAI snapshots.

    The Trotter prefix evolves the state from t=0 to tepai_start_time using
    adaptive Trotterization. Then TE-PAI gates are applied for the remaining
    time, with measurements at each TE-PAI snapshot.
    """
    (n_qubits, signs, gates_arr, gam_list, max_bond, initial_state,
     operator, meas_qubit, trotter_prefix_gates) = run_data

    circ = create_mps_circuit(n_qubits, initial_state, max_bond)

    # Apply Trotter prefix (no measurements needed — those come from the
    # pure-Trotter run which covers the full timeline)
    for prefix_snap_gates in trotter_prefix_gates:
        apply_gates(circ, prefix_snap_gates)

    # Now apply TE-PAI snapshots and measure
    results = []
    # First measurement is at tepai_start_time (gamma_0 = 1 for the TE-PAI segment)
    results.append(measure(circ, operator, meas_qubit))

    for snap_idx, snapshot_gates in enumerate(gates_arr):
        apply_gates(circ, snapshot_gates)
        measured = measure(circ, operator, meas_qubit)
        weight = signs[snap_idx] * gam_list[snap_idx + 1]
        results.append(measured * weight)

    return results


def run_simulation(
    te_pai,
    circuits: List,
    n_qubits: int,
    max_bond: int | None = None,
    operator: str = "X",
    measure_qubit: int = 0,
    initial_state: str = "plus_flip",
    n_workers: int | None = None,
) -> SimulationResult:
    """Run TE-PAI circuits through MPS simulation.

    Each circuit covers the full simulation time T with n_snap measurement
    snapshots built in. Every circuit is an independent run.

    Args:
        te_pai: TE_PAI instance (provides gam_list, T, n_snap, overhead).
        circuits: List of (signs, gates_arr) from te_pai.run_te_pai().
        n_qubits: Number of qubits.
        max_bond: MPS bond dimension limit. None for no truncation.
        operator: Pauli operator to measure ('X', 'Y', 'Z').
        measure_qubit: Qubit index to measure.
        initial_state: Initial state ('plus', 'zero', 'plus_flip').
        n_workers: Parallel workers. None uses sequential execution.

    Returns:
        SimulationResult with times, expectation values, std errors, and raw runs.
    """
    gam_list = te_pai.gam_list
    n_runs = len(circuits)

    run_inputs = [
        (n_qubits, sign, gates_arr, gam_list, max_bond,
         initial_state, operator, measure_qubit)
        for sign, gates_arr in circuits
    ]

    # Execute
    all_results = []
    if n_workers is not None and n_workers > 1 and n_runs > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_single, rd): i for i, rd in enumerate(run_inputs)}
            indexed_results = [None] * n_runs
            for fut in as_completed(futures):
                idx = futures[fut]
                indexed_results[idx] = fut.result()
            all_results = indexed_results
    else:
        for rd in run_inputs:
            all_results.append(_run_single(rd))

    # Compute statistics
    n_snaps = te_pai.n_snap
    n_points = n_snaps + 1  # t=0 + n_snap measurements
    arr = np.array(all_results, dtype=float)

    means = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if n_runs > 1 else np.zeros(n_points)
    stderr = std / np.sqrt(n_runs)

    dT = te_pai.T / te_pai.n_snap
    times = np.arange(n_points) * dT

    return SimulationResult(
        times=times,
        expectation_values=means,
        std_errors=stderr,
        raw_runs=all_results,
        n_circuits=n_runs,
        overhead=te_pai.overhead,
    )


def build_trotter_prefix_gates(hamil, T_total, N_base, n_snap_total, n_snap_prefix):
    """Build Trotter gate lists for the first n_snap_prefix snapshots (adaptive).

    Returns a list of n_snap_prefix sublists, each containing the gates for
    that snapshot interval. Uses the same adaptive-N logic as trotter_mps.
    """
    dT = T_total / n_snap_total
    prefix_gates = []

    for k in range(1, n_snap_prefix + 1):
        N_cumul = N_base * k * k
        N_prev = N_base * (k - 1) * (k - 1)
        N_interval = N_cumul - N_prev

        t_start = (k - 1) * dT
        t_end = k * dT
        interval_dt = t_end - t_start

        steps = np.linspace(t_start, t_end, N_interval, endpoint=False)
        terms_at_t = [hamil.get_term(t) for t in steps]

        snap_gates = []
        for i in range(N_interval):
            for (pauli, ind, coef) in terms_at_t[i]:
                snap_gates.append((pauli, 2 * coef * interval_dt / N_interval, ind))
        prefix_gates.append(snap_gates)

    return prefix_gates


def run_simulation_hybrid(
    te_pai,
    circuits: List,
    trotter_prefix_gates: List,
    n_qubits: int,
    tepai_start_time: float,
    max_bond: int | None = None,
    operator: str = "X",
    measure_qubit: int = 0,
    initial_state: str = "plus_flip",
    n_workers: int | None = None,
) -> SimulationResult:
    """Run hybrid Trotter+TE-PAI circuits through MPS simulation.

    First applies a Trotter prefix (from t=0 to tepai_start_time), then
    applies TE-PAI gates for the remaining time.

    Args:
        te_pai: TE_PAI instance for the TE-PAI portion.
        circuits: List of (signs, gates_arr) from te_pai.run_te_pai().
        trotter_prefix_gates: Gate sublists for the Trotter prefix snapshots.
        n_qubits: Number of qubits.
        tepai_start_time: Time at which TE-PAI takes over from Trotter.
        max_bond: MPS bond dimension limit.
        operator: Pauli operator to measure.
        measure_qubit: Qubit index to measure.
        initial_state: Initial state.
        n_workers: Parallel workers.

    Returns:
        SimulationResult with times starting at tepai_start_time.
    """
    gam_list = te_pai.gam_list
    n_runs = len(circuits)

    run_inputs = [
        (n_qubits, sign, gates_arr, gam_list, max_bond,
         initial_state, operator, measure_qubit, trotter_prefix_gates)
        for sign, gates_arr in circuits
    ]

    all_results = []
    if n_workers is not None and n_workers > 1 and n_runs > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_single_hybrid, rd): i for i, rd in enumerate(run_inputs)}
            indexed_results = [None] * n_runs
            for fut in as_completed(futures):
                idx = futures[fut]
                indexed_results[idx] = fut.result()
            all_results = indexed_results
    else:
        for rd in run_inputs:
            all_results.append(_run_single_hybrid(rd))

    n_snaps = te_pai.n_snap
    n_points = n_snaps + 1
    arr = np.array(all_results, dtype=float)

    means = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if n_runs > 1 else np.zeros(n_points)
    stderr = std / np.sqrt(n_runs)

    dT = te_pai.T / te_pai.n_snap
    times = tepai_start_time + np.arange(n_points) * dT

    return SimulationResult(
        times=times,
        expectation_values=means,
        std_errors=stderr,
        raw_runs=all_results,
        n_circuits=n_runs,
        overhead=te_pai.overhead,
    )


def trotter_mps(hamil, n_qubits, T, N, n_snap, max_bond=None,
                operator="X", measure_qubit=0, initial_state="plus_flip",
                adaptive_N=False):
    """Run a Trotterized time evolution with MPS for reference comparison.

    Args:
        hamil: Hamiltonian instance.
        n_qubits: Number of qubits.
        T: Total simulation time.
        N: Number of Trotter steps. In adaptive mode, this is the number of
            steps for the first snapshot interval dT = T/n_snap.
        n_snap: Number of measurement snapshots.
        max_bond: MPS bond dimension. None for no truncation.
        operator: Pauli operator to measure.
        measure_qubit: Qubit index.
        initial_state: Initial state.
        adaptive_N: If True, scale N with T^2 to keep Trotter error constant.
            N steps are used for the first interval [0, dT]. For snapshot k,
            the cumulative step count is N * k^2, so each successive interval
            uses more steps than the last.

    Returns:
        Tuple of (times, expectation_values) arrays.
    """
    circ = create_mps_circuit(n_qubits, initial_state, max_bond)

    if not adaptive_N:
        steps = np.linspace(0, T, N)
        terms_at_t = [hamil.get_term(t) for t in steps]

        n_per_snap = int(N / n_snap)
        results = [measure(circ, operator, measure_qubit)]

        for i in range(N):
            gates = [
                (pauli, 2 * coef * T / N, ind)
                for (pauli, ind, coef) in terms_at_t[i]
            ]
            apply_gates(circ, gates)

            if (i + 1) % n_per_snap == 0:
                results.append(measure(circ, operator, measure_qubit))
    else:
        N_base = N  # steps for the first interval [0, dT]
        dT = T / n_snap
        results = [measure(circ, operator, measure_qubit)]
        t_evolved = 0.0  # time evolved so far

        for k in range(1, n_snap + 1):
            # Cumulative steps up to snapshot k: N_base * k^2
            N_cumul = N_base * k * k
            # Steps in the previous snapshots
            N_prev = N_base * (k - 1) * (k - 1)
            # Steps for this interval
            N_interval = N_cumul - N_prev

            t_start = (k - 1) * dT
            t_end = k * dT
            interval_dt = t_end - t_start  # = dT

            steps = np.linspace(t_start, t_end, N_interval, endpoint=False)
            terms_at_t = [hamil.get_term(t) for t in steps]

            for i in range(N_interval):
                gates = [
                    (pauli, 2 * coef * interval_dt / N_interval, ind)
                    for (pauli, ind, coef) in terms_at_t[i]
                ]
                apply_gates(circ, gates)

            results.append(measure(circ, operator, measure_qubit))

    times = np.linspace(0, T, n_snap + 1)
    return times, np.array(results)
