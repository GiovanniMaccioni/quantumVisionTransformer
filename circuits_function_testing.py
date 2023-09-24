"""
Testin for Vx circuit.

PRE) Decide the number of qubits for the experiments. This determines the dimension of the vector x and the matrix V.
e.g. For N_QUBITS = 2
1) Initialize at random the "quantum parameter" for V
                  
2) Determine V = |  a   b  |
                 |  c   d  |
    - Load [1., 0.] on the circuit to determine 'a' and 'c'
    - Load [0., 1.] on the circuit to determine 'b' and 'd'

3) Select a 1-norm vector x, run the circuit and verify the result        

#N.B. This test is hardcoded and requires 3 runs of the file in total
"""

from circuits import *
from qiskit import Aer

import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile

import torch

import numpy as np
from qiskit.utils import algorithm_globals

# fix random seeds for reproducibility

np.random.seed(5)
algorithm_globals.random_seed = 5

def get_RBS_parameters(x):
    # get recursively the angles
    def angles(y):
        d = y.shape[-1]
        if d == 2:
            #print(y.shape)
            thetas = torch.acos(y[:,:, 0] / torch.linalg.norm(y, ord=None, dim=2))
            #print("thetas.shape: ", thetas.shape)
            signs = (y[:, :, 1] > 0.).int()
            thetas = signs * thetas + (1. - signs) * (2. * np.pi - thetas)
            #print("thetas.shape: ", thetas.shape)
            thetas = thetas[:,:, None]
            return thetas
        else:
            thetas = torch.acos(torch.linalg.norm(y[:,:, :d//2], ord=None, dim=2, keepdim=True) / torch.linalg.norm(y, ord=None, dim=2, keepdim=True))
            #print("else: thetas.shape: ", thetas.shape)
            #print("y[:,:, :d // 2]", y[:, :, :d // 2])
            thetas_l = angles(y[:, :, :d//2])
            thetas_r = angles(y[:, :, d//2 :])
            thetas = torch.cat((thetas, thetas_l, thetas_r), axis=2)
            #print("thetas.shape: ", thetas.shape)
        return thetas

    # result
    thetas = angles(x)

    return torch.nan_to_num(thetas)

#x = torch.tensor([[torch.sqrt(torch.tensor(2))/2, torch.sqrt(torch.tensor(2))/2]])

"""
Experiment 1: num_qubit 2, matrix quantum paramter 0.75, V = torch.tensor([[0.86586853, 0.15909678],[0.13413147, 0.84090322]])
Experiment 2: num_qubit 4, matrix quantum paramter 0.75, V = torch.tensor([[0.86586853, 0.15909678],[0.13413147, 0.84090322]]) 
"""

x = torch.tensor([[0.5,0.5,0.5,0.5]])
V = torch.tensor([[0.749936, 0.138326, 0.727875, 0.258771],
                  [0.115682, 0.727703, 0.112625, 0.000018],
                  [0.116482, 0.021154, 0.138259, 0.253125],
                  [0.017900, 0.112817, 0.021241, 0.488086]])

param = get_RBS_parameters(x[None, :])

qc = Vx(4, "theta", "sigma")

"""fig = qc._circuit.decompose().draw("mpl", scale = 1)
plt.show()"""

out = torch.matmul(V, x[0])
print(out)

shots = 10000000

temp = torch.tensor([0.75, 0.75, 0.75, 0.75])
all_parameters = torch.cat((temp, param.flatten())).tolist()
bc = qc._circuit.bind_parameters(all_parameters)

"""fig = bc.decompose().draw("mpl", scale = 0.7)
plt.show()"""

backend = Aer.get_backend('aer_simulator')

# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(bc, simulator)

# Run and get counts
result = simulator.run(circ, shots=shots).result()
counts = result.get_counts(circ)
print(counts)
components_values = dict((k, np.sqrt(v/shots)) for k, v in counts.items())
print(components_values)

fig = plot_histogram(counts, title='Bell-State counts')
plt.show()

"""
Possible errors:
    - Circuits
    - Measuring
    - Parameter computation
"""
