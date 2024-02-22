import torch
import torch.nn as nn
import qiskit as qis

import data as d

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from tqdm import tqdm

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from functools import reduce

import numpy as np

import matplotlib.pyplot as plt
from qiskit_algorithms.utils import algorithm_globals

import circuits as C

class HybridMLP(nn.Module):

    def __init__(self, embed_dim, num_layers, num_classes, hidden_size, vec_loader_name, matrix_mul_name):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, embed_dim)
        self.linear3 = nn.Linear(embed_dim, num_classes)

        aersim = AerSimulator(method='statevector', device='GPU')
        sampler = Sampler()
        sampler.set_options(backend=aersim)

        self.vx = C.Vx(embed_dim, vec_loader_name, matrix_mul_name)
        qc, num_weights = self.vx()

        self.num_layers = num_layers
        self.vx_list = nn.ModuleList([TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:], weight_params=qc.parameters[:num_weights], input_gradients=True, sampler=sampler)) for _ in range(num_layers)])

        self.actv = nn.ReLU()
        self.ei = [ 2**j for j in range(0,embed_dim)]

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.actv(x)

        x = x/torch.sqrt(torch.sum(torch.pow(x, 2)+1e-4, dim=1, keepdim=True)+1e-8)
        for i in range(self.num_layers):
            x = self.vx.get_RBS_parameters(x[:,None, :])
            x = self.vx_list[i](x[:,0,:])[:, self.ei]

            x = torch.sqrt(x + 1e-8)
            x = self.actv(x)

        x = self.linear3(x)
        return x