import numpy as np


class Tableau:
    type QubitIdx = int
    type TableauIdx = int
    type MeasurementRes = int

    def __init__(self, n_qubits: QubitIdx):
        self.n_qubits = n_qubits
        self.two_n_qubits = 2 * self.n_qubits
        self.tableau = self._init_tableau()
        self.rng = np.random.default_rng()

    def _init_tableau(self):
        tableau = np.eye(self.two_n_qubits, self.two_n_qubits + 1, dtype=np.bool)
        scratchpad = np.zeros((1, tableau[0].size), dtype=np.bool)
        return np.vstack((tableau, scratchpad), dtype=np.bool)

    def apply_cx(self, a: QubitIdx, b: QubitIdx):
        self._validate_qubit(a)
        self._validate_qubit(b)
        for i in range(self.two_n_qubits):
            self.tableau[i][-1] ^= (
                self.tableau[i][a]
                * self.tableau[i][b + self.n_qubits]
                * (self.tableau[i][b] ^ self.tableau[i][a + self.n_qubits] ^ 1)
            )
            self.tableau[i][b] ^= self.tableau[i][a]
            self.tableau[i][a + self.n_qubits] ^= self.tableau[i][b + self.n_qubits]

    def apply_h(self, a: QubitIdx):
        self._validate_qubit(a)
        for i in range(self.two_n_qubits):
            self.tableau[i][-1] ^= (
                self.tableau[i][a] * self.tableau[i][a + self.n_qubits]
            )
            self.tableau[i][a], self.tableau[i][a + self.n_qubits] = (
                self.tableau[i][a + self.n_qubits],
                self.tableau[i][a],
            )

    def apply_s(self, a: QubitIdx):
        self._validate_qubit(a)
        for i in range(self.two_n_qubits):
            self.tableau[i][-1] ^= (
                self.tableau[i][a] * self.tableau[i][a + self.n_qubits]
            )
            self.tableau[i][a + self.n_qubits] ^= self.tableau[i][a]

    def _g(self, x1: np.int8, z1: np.int8, x2: np.int8, z2: np.int8):
        if x1 == z1 == 0:
            return 0
        if x1 == z1 == 1:
            return z2 - x2
        if x1 == 1 and z1 == 0:
            return z2 * (2 * x2 - 1)
        return x2 * (1 - 2 * z2)

    def _rowsum(self, h: TableauIdx, i: TableauIdx):
        val = 2 * self.tableau[h][-1] + 2 * self.tableau[i][-1]
        for j in range(self.n_qubits):
            val += self._g(
                self.tableau[i][j],
                self.tableau[i][j + self.n_qubits],
                self.tableau[h][j],
                self.tableau[h][j + self.n_qubits],
            )

        self.tableau[h][-1] = val % 4 == 2

        for j in range(self.n_qubits):
            self.tableau[h][j] ^= self.tableau[i][j]
            self.tableau[h][j + self.n_qubits] ^= self.tableau[i][j + self.n_qubits]

    def measure_z(self, a: QubitIdx) -> MeasurementRes:
        self._validate_qubit(a)
        p = None
        for p_cand in range(self.n_qubits, self.two_n_qubits):
            if self.tableau[p_cand][a] == 1:
                p = p_cand
                break

        if p:
            return self._random_z_measurement(a, p)
        return self._deterministic_z_measurement(a)

    def _random_z_measurement(self, a: QubitIdx, p: TableauIdx) -> MeasurementRes:
        for i in range(self.two_n_qubits):
            if i == p or self.tableau[i][a] != 1:
                continue
            self._rowsum(i, p)

        self.tableau[p - self.n_qubits] = self.tableau[p]

        self.tableau[p][:] = 0
        self.tableau[p][a + self.n_qubits] = 1
        self.tableau[p][-1] = self.rng.integers(low=0, high=2)

        return self.tableau[p][-1]

    def _deterministic_z_measurement(self, a: QubitIdx) -> MeasurementRes:
        self.tableau[-1][:] = 0
        for i in range(self.n_qubits):
            if self.tableau[i][a] != 1:
                continue
            self._rowsum(self.two_n_qubits, i + self.n_qubits)

        return self.tableau[-1][-1]

    def _validate_qubit(self, a: QubitIdx):
        if not 0 <= a < self.n_qubits:
            raise ValueError(f"Qubit idx {a} out of range [0..{self.n_qubits}]")

    def __repr__(self):
        return str(self.tableau.view(np.uint8))
