#from qiskit import Aer
import qiskit as qis
import matplotlib.pyplot as plt
from qiskit import transpile

import torch

import numpy as np


from functools import reduce

from gates import RBS_gate, RBS_gate_inverted

    
class Parallel_VectorLoader:
    
    def __init__(self, parameter_vector, inverted=False):
        self.n_qubits = len(parameter_vector)+1
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self._circuit.x(0)

        step = self.n_qubits//2


        id_gate = 0
        temp = self.n_qubits

        idx_list = []
        rbs_list = []

        while temp != 1:
          for i in range(self.n_qubits//temp):
            if inverted:
                rbs_list.append(RBS_gate_inverted(parameter_vector[self.n_qubits-2-id_gate])())
            else:
                rbs_list.append(RBS_gate(parameter_vector[id_gate])())
            
            id_gate += 1

            step = temp // 2
            idx = i*2*step
            idx_list.append([idx, idx+step])

          temp = temp // 2
        
        num_gates = len(rbs_list) - 1
        for i, rbs in enumerate(rbs_list):
            if inverted:
                self._circuit.append(rbs, idx_list[num_gates-i]) 
            else:
                self._circuit.append(rbs, idx_list[i])

        return

    def __call__(self, input_vector=None):
        if input_vector != None:
            parameters = self.get_RBS_parameters(input_vector[:,None, :])
            index = 0
            for param in self._circuit.parameters:
                self._circuit = self._circuit.bind_parameters({param: parameters[index]})
                index += 1
        return self._circuit
    
    
    def get_RBS_parameters(self, x):
        half = x.shape[2]//2
        r1 = []
        r2 = []
        for j in range(0, half):
            r1.append(torch.sqrt(torch.pow(x[:, :, 2*j+1]+1e-4, 2) + torch.pow(x[:, :, 2*j]+1e-4, 2)+1e-8)[:, :, None])
        r1 = torch.cat(r1, dim=2)
        for j in range(0, half-1):
            if j < half//2: 
                r2.insert(0,(torch.sqrt(torch.pow(r1[:, :, half-1-j]+1e-4, 2) + torch.pow(r1[:, :, half-1-j-1]+1e-4, 2))+1e-8)[:, :, None])
            else:
                r2.insert(0, (torch.sqrt(torch.pow(r2[1]+1e-4, 2) + torch.pow(r2[0]+1e-4, 2))+1e-8))
        r2 = torch.cat(r2, dim=2)
        parameters1 = []
        parameters2 = []
        for j in range(0, half):
            mask = (x[:, :, 2*j+1] >= 0.).int().float()
            t1 = (torch.acos(torch.clamp(x[:,:,2*j]/(r1[:,:,j]+1e-8), min=-1., max=1.)))*mask
            t2 = (2*np.pi - torch.acos(torch.clamp(x[:,:,2*j]/(r1[:,:,j]+1e-8), min=-1., max=1.)))*(1. - mask)
            parameters1.append((t1+t2)[:,:,None])
                
        r = torch.cat((r2, r1), dim=2)
        for j in range(0, half-1):
            parameters2.append(torch.acos(torch.clamp(r[:,:,2*j+1]/(r[:,:,j]+1e-8), min=-1., max=1.))[:, :, None])
                

        parameters1 = torch.cat(parameters1, dim=2)
        parameters2 = torch.cat(parameters2, dim=2)
        parameters = torch.cat((parameters2, parameters1), dim=2)

        return parameters
        
    
class Diagonal_VectorLoader:
    
    def __init__(self, parameter_vector, inverted=False):
        # --- Circuit definition ---
        self.n_qubits = len(parameter_vector)+1
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self._circuit.x(0)

        for i in range(self.n_qubits-1):
            if inverted:
                rbs = RBS_gate_inverted(parameter_vector[self.n_qubits-2-i])()
                self._circuit.append(rbs, [self.n_qubits-2-i,self.n_qubits-2+1-i])
            else:
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
                alpha = torch.acos(torch.clamp(x[:, :, i], min=-1., max=1.))
                alpha_list.append(alpha)
            else:
                inv_sin = 1/(torch.sin(alpha_list[i-1]) + 1e-8)
                inv_sin_list.append(inv_sin)
                t = reduce(lambda a, b: a*b, inv_sin_list)
                alpha = torch.acos(torch.clamp(x[:, :, i]*t, min=-1., max=1.))
                alpha_list.append(alpha)
        
        parameters = torch.stack(alpha_list, dim=2)

        return parameters