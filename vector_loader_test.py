from qiskit import Aer
import qiskit as qis
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile

import torch

import numpy as np
from qiskit.utils import algorithm_globals

from functools import reduce

# fix random seeds for reproducibility

np.random.seed(5)
algorithm_globals.random_seed = 5

"""def get_RBS_parameters_parallel(x):
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


def get_RBS_parameters_diagonal(x):
    #
    alpha_list = []
    inv_sin_list = []
    #
    for i in range(x.shape[2] - 1):
        if len(alpha_list) == 0:
            alpha = torch.acos(x[:, :, i])
            #alpha = torch.nan_to_num(alpha)
            alpha_list.append(alpha)
        else:
            inv_sin = 1/torch.sin(alpha_list[i-1])
            #if the sin of the alpha angle is 0, we have to convert the nan value to 0
            inv_sin = torch.nan_to_num(inv_sin)
            inv_sin_list.append(inv_sin)
            t = reduce(lambda a, b: a*b, inv_sin_list)
            alpha = torch.acos(x[:, :, i]*t)
            alpha_list.append(alpha)
    
    parameters = torch.cat(alpha_list, dim=1)
    return torch.nan_to_num(parameters[:,None, :])"""

def rbs_gate_pre():
    matrix = [[1, 0, 0, 0],
            [0, np.cos(np.pi), np.sin(np.pi), 0],
            [0, - np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 0, 1]]
    gate = qis.extensions.UnitaryGate(matrix)

    return gate

def rbs_gate_post(gate, parameter):
    return gate

def rbs_gate(parameter):
   gate = rbs_gate_pre()
   gate = rbs_gate_post(gate, parameter)

   return gate
   

class RBS_gate:
    def __init__(self, parameter):
        self.n_qubits = 2
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self.theta = parameter
        self._circuit.h([0,1])#DEBUG
        self._circuit.cz(0,1)
        self._circuit.ry(self.theta*0.5, 0)
        self._circuit.ry(-self.theta*0.5, 1)
        self._circuit.cz(0,1)
        self._circuit.h([0,1])#DEBUG                                 

    def __call__(self):
      return self._circuit.to_gate()


class Parallel_VectorLoader:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, parameter_vector, input_vector=None):
        # --- Circuit definition ---
        self.n_qubits = len(parameter_vector) + 1
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
            #print(id_gate)
            id_gate += 1
            rbs = rbs_gate(parameter_vector[id_gate-1])
            
            #index = self.n_qubits//temp
            step = temp // 2
            idx = i*2*step
            #print("step:", step)
            #print("idx:", idx, " to idx+1:", idx+step)
            self._circuit.append(rbs, [idx, idx+step])
            #self._circuit.barrier()
          
          #num_gates_each_level = num_gates_each_level*2
          self._circuit.barrier()
          temp = temp // 2
        
        """if input_vector != None:
            parameters = self.get_RBS_parameters(input_vector[:,None, :])
            index = 0
            for param in self._circuit.parameters:
                self._circuit = self._circuit.bind_parameters({param: parameters[index]})
                index += 1"""

    def __call__(self):
      return self._circuit
    
    def get_RBS_parameters(self, x):
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

        torch.nan_to_num(thetas)

        return thetas.flatten().tolist()
        
    
class Diagonal_VectorLoader:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit

    """
    
    def __init__(self, input_vector):
        # --- Circuit definition ---
        self.n_qubits = input_vector.shape[1]
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self._circuit.x(0)

        parameters = self.get_RBS_parameters(input_vector[:,None, :])
        for i in range(self.n_qubits-1):
            rbs = rbs_gate(parameters[i])
            #rbs = RBS_gate(f"theta{i:02d}")
            self._circuit.append(rbs, [i, i+1])

    def __call__(self):
      return self._circuit
    
    def get_RBS_parameters(self, x):
        #
        alpha_list = []
        inv_sin_list = []
        #
        for i in range(x.shape[2] - 1):
            if len(alpha_list) == 0:
                alpha = torch.acos(x[:, :, i])
                #alpha = torch.nan_to_num(alpha)
                alpha_list.append(alpha)
            else:
                inv_sin = 1/torch.sin(alpha_list[i-1])
                #if the sin of the alpha angle is 0, we have to convert the nan value to 0
                inv_sin = torch.nan_to_num(inv_sin)
                inv_sin_list.append(inv_sin)
                t = reduce(lambda a, b: a*b, inv_sin_list)
                alpha = torch.acos(x[:, :, i]*t)
                alpha_list.append(alpha)
        
        parameters = torch.cat(alpha_list, dim=1)
        torch.nan_to_num(parameters[:,None, :])

        return parameters.flatten().tolist()
    
class Butterfly:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, num_features):
        # --- Circuit definition ---
        self.n_qubits = num_features
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self.thetas = [] 
        number_of_parameters = (self.n_qubits//2 * np.log2(self.n_qubits)).astype(int)
        #self.num_gates_each_level = self.num_qubits//2
        self.num_gates = self.n_qubits//2 * np.log2(self.n_qubits)
        temp = self.n_qubits
        #print((self.n_qubits//temp) != self.n_qubits)
        id_gate = 0

        parameters = torch.randn((number_of_parameters,)).tolist()

        
        while (self.n_qubits//temp) != self.n_qubits:
          #print(temp)
          for i in range(self.n_qubits//2):
            #print(id_gate)
            rbs = rbs_gate(parameters[id_gate])
            id_gate += 1
            #index = self.n_qubits//temp
            step = temp // 2
            idx = i + (i//step)*step
            #print("step:", step)
            #print("idx:", idx, " to idx+1:", idx+step)
            self._circuit.append(rbs, [idx, idx+step])
            
          self._circuit.barrier()
          temp = temp // 2

    def __call__(self):
      return self._circuit

#x = torch.tensor([[torch.sqrt(torch.tensor(2))/2, torch.sqrt(torch.tensor(2))/2]])

#x = torch.tensor([[0.5,0.5,0.5,0.5]])
#x = torch.tensor([[0.,1.,0.,0.]])
#x = torch.tensor([[1/torch.sqrt(torch.tensor(2)),1/torch.sqrt(torch.tensor(2))]])
#x = torch.tensor([[0.,0.,0.,0.,0.,0.,0.,1.]])
#x = torch.tensor([[0.5,0.5, 0.5, 0.5]])
x = torch.tensor([[0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25, 0.25,0.25,0.25,0.25]])
#x = torch.tensor([[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8, 0.9,0.10,0.11,0.12, 0.13,0.14,0.15,0.16]])
#x = torch.tensor([[7/torch.sqrt(torch.tensor(113)), 8/torch.sqrt(torch.tensor(113))]])
#x = torch.tensor([[1.,0.]])

"""param = get_RBS_parameters_diagonal(x[:, None, :])#x[None, :])
param = param.flatten().tolist()"""

"""temp = param[0]
param[0] = param[2]
param[2] = param[1]
param[1] = temp
sub = [np.pi/2, np.pi/2, np.pi/2]"""

"""for i in range(len(param)):
   param[i] = param[i] - sub[i]"""



"""vec_loader = VectorLoader(2, "theta")(param)
vec_loader.measure_all()
"""
"""
Diagonal Vector Loader
"""


"""#TODO Find the algorithm for parameter computation

#Diagonal Vector Loader
vec_loader = qis.QuantumCircuit(n_qubits)
vec_loader.x(0)
for i in range(n_qubits-1):
    rbs = rbs_gate(param[i])
    #rbs = RBS_gate(f"theta{i:02d}")
    vec_loader.append(rbs, [i, i+1])
vec_loader.measure_all()"""


"""cir0 = qis.QuantumCircuit(1)
cir0.ry(0.5, 0)

theta = qis
cir1= qis.QuantumCircuit(1)
theta = qis.circuit.Parameter("theta")
cir1.ry(theta, 0)"""


"""thetas = qis.circuit.ParameterVector("thetas", length=15)

circuit = Parallel_VectorLoader(thetas, x)()
circuit.measure_all()"""

matrix = Butterfly(16)()

circuit = qis.QuantumCircuit(16)
#self._circuit.h([0,1, 2, 3])#DEBUG
circuit.compose(vec_loader.compose(matrix), inplace = True) 
#self._circuit.h([0,1, 2, 3])#DEBUG
#TODO take the z expectaion value _circuit.exp_val<-----
circuit.measure_all()

shots = 100000

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

"""
Possible errors:
    - Circuits
    - Measuring
    - Parameter computation
"""




