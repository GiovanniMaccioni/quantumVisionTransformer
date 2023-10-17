from qiskit import Aer
import qiskit as qis
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit import transpile

import torch

import numpy as np
from qiskit.utils import algorithm_globals

from functools import reduce

from gates import RBS_gate

    
class Parallel_VectorLoader:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, parameter_vector):
        # --- Circuit definition ---
        self.n_qubits = len(parameter_vector)+1
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

        #parameters = self.get_RBS_parameters(input_vector[:,None, :])
        
        #while num_gates_each_level != (self.n_qubits//2):#FIXME doesn't work with num_features == 2
        while temp != 1:#FIXME doesn't work with num_features == 2
          #print(temp)
          for i in range(self.n_qubits//temp):
            #print(id_gate)
            id_gate += 1
            rbs = RBS_gate(parameter_vector[id_gate - 1])()
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

    def __call__(self, input_vector=None):
        if input_vector != None:
            parameters = self.get_RBS_parameters(input_vector[:,None, :])
            index = 0
            for param in self._circuit.parameters:
                self._circuit = self._circuit.bind_parameters({param: parameters[index]})
                index += 1
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
    
    def __init__(self, parameter_vector):
        # --- Circuit definition ---
        self.n_qubits = len(parameter_vector)+1
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self._circuit.x(0)

        for i in range(self.n_qubits-1):
            rbs = RBS_gate(parameter_vector[i])()
            self._circuit.append(rbs, [i, i+1])

    def __call__(self, input_vector=None):
        if input_vector != None:
            parameters = self.get_RBS_parameters(input_vector[:,None, :])
            index = 0
            for param in self._circuit.parameters:
                self._circuit = self._circuit.bind_parameters({param: parameters[index]})
                index += 1
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