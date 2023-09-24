from qiskit import Aer
import qiskit as qis
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
        #Get the len of the components. In the first call it is the length of the vector in input!!
        d = y.shape[-1]
        if d == 2:
            #print(y.shape)
            thetas = torch.acos(y[:,:, 0] / torch.linalg.norm(y, ord=None, dim=2))
            #print("thetas.shape: ", thetas.shape)
            signs = (y[:, :, 1] > 0.).int()
            thetas = signs * thetas + (1. - signs) * (2*np.pi - thetas)
            #print("thetas.shape: ", thetas.shape)
            thetas = thetas[:,:, None]
            return thetas
        else:
            #First d/2 values, 
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

def get_RBS_parameters2(x):
    # get recursively the angles
    def angles(y):
        #Get the len of the components. In the first call it is the length of the vector in input!!
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
            #First d/2 values, 
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

class RBS_gate:
    def __init__(self, parameter_name):
        self.n_qubits = 2
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self.theta = qis.circuit.Parameter(parameter_name)
        self._circuit.h([0,1])#DEBUG
        self._circuit.cz(0,1)
        self._circuit.ry(self.theta*0.5, 0)
        self._circuit.ry(-self.theta*0.5, 1)
        self._circuit.cz(0,1)
        self._circuit.h([0,1])#DEBUG

    def __call__(self, theta=None):
      if theta != None:
        self._circuit = self._circuit.bind_parameters({self.theta: theta})
      
      return self._circuit.to_gate()

class VectorLoader:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, num_features, parameter_name):
        # --- Circuit definition ---
        self.n_qubits = num_features
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self._circuit.x(0)

        #self.num_gates = self.n_qubits//2 * torch.log2(self.n_qubits)
        #self.num_gates_each_level = self.num_qubits//2
        #self.num_gates = self.n_qubits//2 * np.log2(self.n_qubits)
        step = self.n_qubits//2
        #print((self.n_qubits//temp) != self.n_qubits)
        num_gates_each_level = 1
        id_gate = 0
        temp = self.n_qubits
        
        #while num_gates_each_level != (self.n_qubits//2):#FIXME doesn't work with num_features == 2
        while temp != 1:#FIXME doesn't work with num_features == 2
          #print(temp)
          for i in range(self.n_qubits//temp):
            parameter = parameter_name + "{:02d}"
            parameter = parameter.format(id_gate)
            #print(id_gate)
            id_gate += 1
            rbs = RBS_gate(parameter)
            #index = self.n_qubits//temp
            step = temp // 2
            idx = i*2*step
            #print("step:", step)
            #print("idx:", idx, " to idx+1:", idx+step)
            self._circuit.append(rbs(), [idx, idx+step])
            #self._circuit.barrier()
          
          #num_gates_each_level = num_gates_each_level*2
          self._circuit.barrier()
          temp = temp // 2

    def __call__(self, thetas=None):
      if thetas != None:
        #all_thetas = [i for i in range(len(thetas))]
        #print(thetas)
        index = 0
        for param in self._circuit.parameters:
          #print(len(self._circuit.parameters))
          self._circuit = self._circuit.bind_parameters({param: thetas[index]})
          index += 1
    
      return self._circuit
    
class VectorLoader:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, num_features, parameter_name):
        # --- Circuit definition ---
        self.n_qubits = num_features
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self._circuit.x(0)

        #self.num_gates = self.n_qubits//2 * torch.log2(self.n_qubits)
        #self.num_gates_each_level = self.num_qubits//2
        #self.num_gates = self.n_qubits//2 * np.log2(self.n_qubits)
        step = self.n_qubits//2
        #print((self.n_qubits//temp) != self.n_qubits)
        num_gates_each_level = 1
        id_gate = 0
        temp = self.n_qubits
        
        #while num_gates_each_level != (self.n_qubits//2):#FIXME doesn't work with num_features == 2
        while temp != 1:#FIXME doesn't work with num_features == 2
          #print(temp)
          for i in range(self.n_qubits//temp):
            parameter = parameter_name + "{:02d}"
            parameter = parameter.format(id_gate)
            #print(id_gate)
            id_gate += 1
            rbs = RBS_gate(parameter)
            #index = self.n_qubits//temp
            step = temp // 2
            idx = i*2*step
            #print("step:", step)
            #print("idx:", idx, " to idx+1:", idx+step)
            self._circuit.append(rbs(), [idx, idx+step])
            #self._circuit.barrier()
          
          #num_gates_each_level = num_gates_each_level*2
          self._circuit.barrier()
          temp = temp // 2

    def __call__(self, thetas=None):
      if thetas != None:
        #all_thetas = [i for i in range(len(thetas))]
        #print(thetas)
        index = 0
        for param in self._circuit.parameters:
          #print(len(self._circuit.parameters))
          self._circuit = self._circuit.bind_parameters({param: thetas[index]})
          index += 1
    
      return self._circuit

#x = torch.tensor([[torch.sqrt(torch.tensor(2))/2, torch.sqrt(torch.tensor(2))/2]])

x = torch.tensor([[0.,0.,0.,1.]])
#x = torch.tensor([[0.,0.,0.,0.,0.,0.,0.,1.]])
#x = torch.tensor([[0.5,0.5, 0.5, 0.5]])
#x = torch.tensor([[0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25]])
#x = torch.tensor([[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8, 0.9,0.10,0.11,0.12, 0.13,0.14,0.15,0.16]])
#x = torch.tensor([[7/torch.sqrt(torch.tensor(113)), 8/torch.sqrt(torch.tensor(113))]])
#x = torch.tensor([[0.,1.]])

param = get_RBS_parameters(x[None, :])
param = param.flatten().tolist()

print(param)

temp = param[0]
param[0] = param[2]
param[2] = param[1]
param[1] = temp
sub = [np.pi/2, np.pi/2, np.pi/2]

print(param)
n_qubits = len(param)+1

for i in range(len(param)):
   param[i] = param[i] - sub[i]

print(param)



"""vec_loader = VectorLoader(2, "theta")(param)
vec_loader.measure_all()
"""
"""
Diagonal Vector Loader
"""

def rbs_gate(parameter):
    matrix = [[1, 0, 0, 0],
            [0, np.cos(parameter), np.sin(parameter), 0],
            [0, - np.sin(parameter), np.cos(parameter), 0],
            [0, 0, 0, 1]]
    gate = qis.extensions.UnitaryGate(matrix)

    return gate


vec_loader = qis.QuantumCircuit(n_qubits)
vec_loader.x(0)
for i in range(n_qubits-1):
    rbs = rbs_gate(param[i])
    #rbs = RBS_gate(f"theta{i:02d}")
    vec_loader.append(rbs, [i, i+1])
vec_loader.measure_all()


shots = 1000000

fig = vec_loader.draw("mpl", scale = 0.5)
plt.show()

fig = vec_loader.decompose().draw("mpl", scale = 0.5)
plt.show()

# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
circ = transpile(vec_loader, simulator)#<----

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




