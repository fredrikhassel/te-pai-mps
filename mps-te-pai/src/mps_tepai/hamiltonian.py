"""Hamiltonian definitions for TE-PAI simulations."""

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import integrate
from typing import Callable, List, Tuple


@dataclass
class Hamiltonian:
    nqubits: int
    terms: List[Tuple[str, List[int], Callable[[float], float]]]

    @staticmethod
    def spin_chain(n, freqs, j=0.1):
        def J(t):
            return j # Time-dependent coupling can be added here if desired
        terms = [
            (gate, [k, (k + 1) % n], J)
            for k, gate in product(range(n), ["XX", "YY", "ZZ"])
        ]
        terms += [("Z", [k], lambda t, k=k: freqs[k]) for k in range(n)]
        return Hamiltonian(n, terms)

    def get_term(self, t):
        """Return terms with coefficients evaluated at time t."""
        return [(term[0], term[1], term[2](t)) for term in self.terms]

    def coefs(self, t: float):
        """Return list of coefficient values at time t."""
        return [term[2](t) for term in self.terms]

    def l1_norm(self, T: float):
        """Integrate the L1 norm of coefficients from 0 to T."""
        fn = lambda t: np.linalg.norm(self.coefs(t), 1)
        return integrate.quad(fn, 0, T, limit=100)[0]

    def __len__(self):
        return len(self.terms)
