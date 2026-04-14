"""Example: Compare Trotterized time-evolution vs MPS TE-PAI estimate.

Simulates a 10-qubit spin chain measuring <X_0> over time.
Prints both the Trotter reference and the TE-PAI estimate side by side.
"""
import sys
import os

# Add package to path for running without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    import numpy as np
    from mps_tepai import Hamiltonian, TE_PAI, trotter_mps, run_simulation

    # --- Parameters ---
    n_qubits = 10
    T = 1.0                 # total time
    dt = 0.1                # time step between measurements
    N_trotter = 1000        # Trotter steps per segment (high accuracy reference)
    N_tepai = 100           # Trotter steps per dT for TE-PAI decomposition
    delta = np.pi / 516     # PAI rotation angle
    n_circuits = 1000       # number of TE-PAI circuits
    max_bond = 16           # MPS bond dimension
    seed = 0                # random seed for reproducibility

    # --- Build Hamiltonian ---
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(-1, 1, size=n_qubits)
    hamil = Hamiltonian.spin_chain(n_qubits, freqs, j=0.1)

    # --- 1. Trotter reference (high N for accuracy) ---
    print("Running Trotter reference simulation...")
    n_snap = int(round(T / dt))
    trotter_times, trotter_vals = trotter_mps(
        hamil, n_qubits, T=T, N=N_trotter, n_snap=n_snap,
        max_bond=max_bond, operator="X", measure_qubit=0,
        initial_state="plus_flip",
    )
    # Convert from [0,1] to [-1,1] expectation values
    trotter_expect = 2 * trotter_vals - 1

    # --- 2. TE-PAI simulation ---
    # Each circuit covers the full time T with n_snap measurement snapshots.
    print(f"Generating {n_circuits} TE-PAI circuits...")
    te_pai = TE_PAI(hamil, n_qubits, delta=delta, T=T, N=N_tepai, n_snap=n_snap)
    circuits = te_pai.run_te_pai(n_circuits, seed=seed)

    print(f"Running MPS TE-PAI simulation (max_bond={max_bond})...")
    result = run_simulation(
        te_pai=te_pai,
        circuits=circuits,
        n_qubits=n_qubits,
        max_bond=max_bond,
        operator="X",
        measure_qubit=0,
        initial_state="plus_flip",
        n_workers=2,
    )
    tepai_expect = 2 * result.expectation_values - 1
    tepai_stderr = 2 * result.std_errors

    # --- Print results ---
    print("\n" + "=" * 70)
    print(f"  {'Time':>6s}  |  {'Trotter <X_0>':>14s}  |  {'TE-PAI <X_0>':>14s}  |  {'Std Error':>10s}")
    print("-" * 70)

    n_print = min(len(trotter_times), len(result.times))
    for i in range(n_print):
        t = trotter_times[i]
        tr_val = trotter_expect[i]
        tp_val = tepai_expect[i] if i < len(tepai_expect) else float("nan")
        tp_err = tepai_stderr[i] if i < len(tepai_stderr) else float("nan")
        print(f"  {t:6.2f}  |  {tr_val:14.6f}  |  {tp_val:14.6f}  |  {tp_err:10.6f}")

    print("=" * 70)

    # --- Gate count comparison ---
    trotter_gates = N_trotter * len(hamil)  # one gate per term per Trotter step
    tepai_gates_per_circuit = []
    for _signs, gates_arr in circuits:
        cumulative = [0]
        running = 0
        for snap_gates in gates_arr:
            running += len(snap_gates)
            cumulative.append(running)
        tepai_gates_per_circuit.append(cumulative)
    tepai_gate_counts = np.mean(tepai_gates_per_circuit, axis=0)

    print(f"\nTrotter gates (total):           {trotter_gates}")
    print(f"TE-PAI gates/circuit: {tepai_gate_counts[-1]:.1f}")
    print(f"Gate ratio (Trotter / TE-PAI):   {trotter_gates / tepai_gate_counts[-1]:.1f}x")
    print(f"\nTE-PAI overhead (gamma): {te_pai.overhead:.4f}")
    print(f"Circuits (independent runs): {result.n_circuits}")

    # --- Plot ---
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(trotter_times, trotter_expect, color="black", label="Trotter")
    ax.plot(result.times, tepai_expect, color="tab:green", label="TE-PAI")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle X_0 \rangle$")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
