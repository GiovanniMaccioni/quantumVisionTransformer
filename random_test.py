from qiskit import Aer
import qiskit as qis
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile

import torch

circ = qis.QuantumCircuit(3)

# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)

fig = circ.draw("mpl")
plt.show()

circ.measure_all()

fig = circ.draw("mpl")
plt.show()

# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)#<----

# Run and get counts
result = simulator.run(circ, shots=10000).result()
counts = result.get_counts(circ)

fig = plot_histogram(counts, title='Bell-State counts')
plt.show()

