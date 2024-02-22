import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit as qis
from qiskit.visualization import *
from qiskit_algorithms.utils import algorithm_globals

import vector_loaders
import orthogonal_layers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Vx:
    """ 
    This class implements the Matrix-Vector multiplication
    """
    
    def __init__(self, n_qubits, vec_loader_name, ort_name):
        thetas = qis.circuit.ParameterVector("thetas", length=n_qubits-1)

        if vec_loader_name == 'diagonal':
          self.vec_loader = vector_loaders.Diagonal_VectorLoader(thetas)
        elif vec_loader_name == 'parallel':
          self.vec_loader = vector_loaders.Parallel_VectorLoader(thetas)
        else:
          self.vec_loader = vector_loaders.Parallel_VectorLoader(thetas)
           
        vec_loader = self.vec_loader()

        if ort_name == 'pyramid':
           matrix, self.num_weights = orthogonal_layers.Pyramid(n_qubits)()
        elif ort_name == 'butterfly':
          matrix, self.num_weights = orthogonal_layers.Butterfly(n_qubits)()
        else:
           matrix, self.num_weights = orthogonal_layers.Pyramid(n_qubits)()
        

        self._circuit = qis.QuantumCircuit(n_qubits)
        self._circuit.compose(vec_loader.compose(matrix), inplace = True) 
        self._circuit.measure_all()
        # ---------------------------
    def __call__(self):
      return self._circuit, self.num_weights
    
    def get_RBS_parameters(self, x):
        return self.vec_loader.get_RBS_parameters(x)


class xWx:
    """ 
    This class implements the Vector-Matrix-Vector multiplication
    """
    
    def __init__(self, n_qubits, vec_loader_name, ort_name):
        self.num_qubits = n_qubits
        thetas = qis.circuit.ParameterVector("thetas", length=n_qubits-1)
        phis = qis.circuit.ParameterVector("phis", length=n_qubits-1)

        self._circuit = qis.QuantumCircuit(n_qubits)
        if vec_loader_name == 'diagonal':
          self.vec_loader = vector_loaders.Diagonal_VectorLoader(thetas)
          vec_loader_adjoint = vector_loaders.Diagonal_VectorLoader(phis, True)()
          vec_loader = self.vec_loader()
        elif vec_loader_name == 'parallel':
          self.vec_loader = vector_loaders.Parallel_VectorLoader(thetas)
          vec_loader_adjoint = vector_loaders.Parallel_VectorLoader(phis, True)()
          vec_loader = self.vec_loader()
        else:
          self.vec_loader = vector_loaders.Diagonal_VectorLoader(thetas)
          vec_loader_adjoint = vector_loaders.Diagonal_VectorLoader(phis, True)()
          vec_loader = self.vec_loader()    

        if ort_name == 'pyramid':
          w, self.num_weights = orthogonal_layers.Pyramid(n_qubits)()
        elif ort_name == 'butterfly':
          w, self.num_weights = orthogonal_layers.Butterfly(n_qubits)()
        else:
          w, self.num_weights = orthogonal_layers.Pyramid(n_qubits)()


        self._circuit.compose(vec_loader.compose(w.compose(vec_loader_adjoint)), inplace = True)  
        self._circuit.measure_all()
        # ---------------------------
    def __call__(self):
      return self._circuit, self.num_weights
    
    def get_RBS_parameters(self, x):
        return self.vec_loader.get_RBS_parameters(x)
