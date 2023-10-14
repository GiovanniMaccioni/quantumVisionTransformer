from qiskit import Aer
import qiskit as qis
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile

import torch

import numpy as np
from qiskit.utils import algorithm_globals

def rbs_gate(parameter):
    matrix = [[1, 0, 0, 0],
            [0, np.cos(parameter), np.sin(parameter), 0],
            [0, - np.sin(parameter), np.cos(parameter), 0],
            [0, 0, 0, 1]]
    gate = qis.extensions.UnitaryGate(matrix)

    return gate

rbs_test = qis.QuantumCircuit(2)

parameter = np.pi/2

rbs = rbs_gate(parameter)
    #rbs = RBS_gate(f"theta{i:02d}")
rbs_test.append(rbs, [0, 1])
rbs_test.measure_all()


fig = rbs_test.draw("mpl", scale = 1.0)
plt.show()

fig = rbs_test.decompose().draw("mpl", scale = 1.0)
plt.show()

shots = 100000

simulator = Aer.get_backend('aer_simulator')
circ = transpile(rbs_test, simulator)#<----

# Run and get counts
result = simulator.run(circ, shots=shots).result()
counts = result.get_counts(circ)
print(counts)
components_values = dict((k, np.sqrt(v/shots)) for k, v in counts.items())
print(components_values)

fig = plot_histogram(counts, title='Bell-State counts')
plt.show()