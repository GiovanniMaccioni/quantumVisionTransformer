import qiskit as qis
import numpy as np

class RBS_gate:
    def __init__(self, parameter):
        self.n_qubits = 2
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self.theta = parameter
        self._circuit.h([0,1])#DEBUG
        self._circuit.cz(0,1)
        self._circuit.ry(self.theta, 0)
        self._circuit.ry(-self.theta, 1)
        self._circuit.cz(0,1)
        self._circuit.h([0,1])#DEBUG                                 

    def __call__(self):
      return self._circuit.to_gate()