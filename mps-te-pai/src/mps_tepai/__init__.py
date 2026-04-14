"""MPS-based Time-Evolution via Probabilistic Approximate Identity (TE-PAI)."""

from mps_tepai.hamiltonian import Hamiltonian
from mps_tepai.circuit_gen import TE_PAI
from mps_tepai.mps_runner import (
    SimulationResult,
    run_simulation,
    run_simulation_hybrid,
    build_trotter_prefix_gates,
    trotter_mps,
    create_mps_circuit,
    apply_gates,
    measure,
)
import numpy as np


def simulate(
    n_qubits: int,
    T: float,
    dt: float,
    delta: float,
    n_circuits: int,
    N: int = 100,
    j: float = 0.1,
    max_bond: int | None = None,
    operator: str = "X",
    measure_qubit: int = 0,
    initial_state: str = "plus_flip",
    hamiltonian: str | Hamiltonian = "spin_chain",
    freqs: np.ndarray | None = None,
    seed: int = 0,
    n_workers: int | None = None,
) -> SimulationResult:
    """One-call interface: define Hamiltonian, generate circuits, run MPS simulation.

    Args:
        n_qubits: Number of qubits.
        T: Total simulation time.
        dt: Time step between measurement snapshots.
        delta: PAI rotation angle (e.g. np.pi/64).
        n_circuits: Number of full-length TE-PAI circuits (= independent samples).
        N: Total Trotter steps (default 100).
        j: Coupling strength for built-in Hamiltonians.
        max_bond: MPS bond dimension limit. None for no truncation.
        operator: Pauli operator to measure ('X', 'Y', 'Z').
        measure_qubit: Qubit index to measure.
        initial_state: 'plus', 'zero', or 'plus_flip'.
        hamiltonian: 'spin_chain', 'nnn', '2d', or a Hamiltonian instance.
        freqs: On-site Z-field frequencies. None for random (seeded).
        seed: Random seed for frequency generation.
        n_workers: Number of parallel workers.

    Returns:
        SimulationResult with times, expectation_values, std_errors, and raw_runs.
    """
    rng = np.random.default_rng(seed)
    if freqs is None:
        freqs = rng.uniform(-1, 1, size=n_qubits)

    if isinstance(hamiltonian, Hamiltonian):
        hamil = hamiltonian
    elif hamiltonian == "spin_chain":
        hamil = Hamiltonian.spin_chain(n_qubits, freqs, j=j)
    elif hamiltonian == "nnn":
        hamil = Hamiltonian.next_nearest_neighbor(n_qubits, freqs)
    elif hamiltonian == "2d":
        hamil = Hamiltonian.lattice_2d(n_qubits, J=j, freqs=freqs)
    else:
        raise ValueError(f"Unknown hamiltonian type: {hamiltonian}")

    n_snap = int(round(T / dt))

    te_pai = TE_PAI(hamil, n_qubits, delta=delta, T=T, N=N, n_snap=n_snap)
    circuits = te_pai.run_te_pai(n_circuits, n_workers=n_workers)

    return run_simulation(
        te_pai=te_pai,
        circuits=circuits,
        n_qubits=n_qubits,
        max_bond=max_bond,
        operator=operator,
        measure_qubit=measure_qubit,
        initial_state=initial_state,
        n_workers=n_workers,
    )


__all__ = [
    "Hamiltonian",
    "TE_PAI",
    "SimulationResult",
    "simulate",
    "run_simulation",
    "run_simulation_hybrid",
    "build_trotter_prefix_gates",
    "trotter_mps",
    "create_mps_circuit",
    "apply_gates",
    "measure",
]
