"""TE-PAI circuit generation (no file I/O)."""

from dataclasses import dataclass
import numpy as np

from mps_tepai.pai import gamma, prob_list
from mps_tepai.sampling import batch_sampling


@dataclass
class TE_PAI:
    """Generate TE-PAI randomized circuits for a given Hamiltonian.

    Attributes:
        nq: Number of qubits.
        delta: PAI rotation angle.
        T: Total simulation time per segment.
        N: Number of Trotter steps.
        n_snap: Number of measurement snapshots.
        overhead: Sampling overhead factor.
        gamma_final: Final gamma normalization.
        gam_list: Per-snapshot cumulative gamma values.
        expected_num_gates: Expected gate count per circuit.
    """

    def __init__(self, hamil, n_qubits, delta, T, N, n_snap):
        self.nq = n_qubits
        self.n_snap = n_snap
        self.delta = delta
        self.T = T
        self.L = len(hamil)

        # N is Trotter steps per snapshot segment (matching original implementation).
        # Total steps across the full time T is N * n_snap.
        N_total = N * n_snap
        self.N = N_total

        steps = np.linspace(0, T, N_total)
        angles = [[2 * np.abs(coef) * T / N_total for coef in hamil.coefs(t)] for t in steps]
        n = int(N_total / n_snap)  # = N (steps per segment)

        self.gam_list = [1] + [
            np.prod([gamma(angles[j], delta) for j in range((i + 1) * n)])
            for i in range(n_snap)
        ]
        self.gamma_final = self.gam_list[-1] if N_total > 0 else 0

        self.probs = [prob_list(angles[i], delta) for i in range(N_total)]
        self.terms = [hamil.get_term(t) for t in steps]
        self.overhead = np.exp(2 * hamil.l1_norm(T) * np.tan(delta / 2))
        self.expected_num_gates = ((3 - np.cos(delta)) / np.sin(delta)) * hamil.l1_norm(T)

    def run_te_pai(self, num_circuits, n_workers=None, seed=None):
        """Generate TE-PAI circuits.

        Args:
            num_circuits: Number of randomized circuits to generate.
            n_workers: Number of parallel workers for sampling.
            seed: Optional int seed for reproducible sampling.

        Returns:
            List of (signs, gates_arr) tuples where:
              - signs: list of n_snap cumulative sign factors (+1 or -1),
                       one per snapshot (tracks pi-parity up to that point)
              - gates_arr: list of snapshots, each a list of (pauli, angle, qubits) gates
        """
        if num_circuits <= 0:
            return []

        indices = batch_sampling(np.array(self.probs), num_circuits, n_workers, seed=seed)
        circuits = []
        for idx in indices:
            sign, gates_arr = self._gen_circuit(idx)
            circuits.append((sign, gates_arr))
        return circuits

    def _gen_circuit(self, index):
        """Generate a single randomized circuit from sampled indices.

        Returns (signs, gates_arr) where signs[k] is the cumulative
        pi-parity (+1 or -1) after applying all gates through snapshot k.
        """
        gates_arr = []
        signs = []
        sign = 1
        n = int(self.N / self.n_snap)

        for i, inde in enumerate(index):
            if i % n == 0:
                gates_arr.append([])
            for j, val in inde:
                pauli, ind, coef = self.terms[i][j]
                if val == 3:
                    sign *= -1
                    gate = (pauli, np.pi, ind)
                else:
                    gate = (pauli, np.sign(coef) * self.delta, ind)
                gates_arr[-1].append(gate)
            # Record cumulative sign at each snapshot boundary
            if (i + 1) % n == 0:
                signs.append(sign)

        return signs, gates_arr

    def sample_num_gates(self, n):
        """Sample n circuits and return their gate counts."""
        res = batch_sampling(np.array(self.probs), n)
        return [sum(len(r) for r in re) for re in res]
