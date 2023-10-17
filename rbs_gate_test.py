from qiskit import Aer
import qiskit as qis
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile

import torch

import numpy as np
from qiskit.utils import algorithm_globals

from functools import reduce
import vector_loaders                              

shots = 10000

x = torch.tensor([[0.5,0.5,0.5,0.5]])
thetas = qis.circuit.ParameterVector("thetas", length=3)
    
circuit = vector_loaders.Diagonal_VectorLoader(thetas)(x)
circuit.measure_all()

fig = circuit.draw("mpl", scale = 1.0)
plt.show()

fig = circuit.decompose().draw("mpl", scale = 1.0)
plt.show()

# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(circuit, simulator)#<----

# Run and get counts
result = simulator.run(circ, shots=shots).result()
counts = result.get_counts(circ)
print(counts)
components_values = dict((k, np.sqrt(v/shots)) for k, v in counts.items())
print(components_values)

fig = plot_histogram(counts, title='Bell-State counts')
plt.show()