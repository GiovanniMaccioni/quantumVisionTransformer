import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit as qis
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit_aer import AerSimulator
from qiskit import Aer

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import Sampler as AerSampler

import utils as ut

import vector_loaders
import matrix_multiplication

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
from functools import reduce

def get_RBS_parameters(x):
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
        
        parameters = torch.stack(alpha_list, dim=2)
        parameters = torch.nan_to_num(parameters)

        return parameters


class Vx:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits):
        # --- Circuit definition ---
        thetas = qis.circuit.ParameterVector("thetas", length=n_qubits-1)
        vec_loader = vector_loaders.Diagonal_VectorLoader(thetas)()
        matrix = matrix_multiplication.Butterfly(n_qubits)()

        self._circuit = qis.QuantumCircuit(n_qubits)
        #self._circuit.h([0,1, 2, 3])#DEBUG
        self._circuit.compose(vec_loader.compose(matrix), inplace = True) 
        #self._circuit.h([0,1, 2, 3])#DEBUG
        #TODO take the z expectaion value _circuit.exp_val<-----
        self._circuit.measure_all()
        # ---------------------------
    def __call__(self):
      return self._circuit


class xWx:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits):
        # --- Circuit definition ---
        self.num_qubits = n_qubits ## Aggiunto per compatibilità con EstimatorQNN
        thetas = qis.circuit.ParameterVector("thetas", length=n_qubits-1)
        phis = qis.circuit.ParameterVector("phis", length=n_qubits-1)

        self._circuit = qis.QuantumCircuit(n_qubits)
        self.vec_loader = vector_loaders.Diagonal_VectorLoader(thetas)()
        self.W = matrix_multiplication.Butterfly(n_qubits)()
        self.vec_loader_adjoint = vector_loaders.Diagonal_VectorLoader(phis)().inverse()

        self._circuit.compose(self.vec_loader.compose(self.W.compose(self.vec_loader_adjoint)), inplace = True)
        #TODO take the z expectaion value _circuit.exp_val<-----   
        self._circuit.measure_all()
        # ---------------------------
    def __call__(self):
      return self._circuit


class quantumAttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim

        self.num_patches = num_patches + 1#+1 because it takes into account the class vector
        
        """
        #simulator_gpu = Aer.get_backend('aer_simulator')
        simulator_gpu = Aer.get_backend('aer_simulator_statevector')
        simulator_gpu.set_options(device='GPU')

        self.quantum_instance = qis.utils.QuantumInstance(simulator_gpu,
                    # we'll set a seed for reproducibility
                    shots = 1024, seed_simulator = 2718,
                    seed_transpiler = 2718)
        """
        seed = 170
        self.sampler = AerSampler(
            run_options={"seed": seed, "shots": 1024},
            transpile_options={"seed_transpiler": seed}
        )

        #Missing initialization of weights before TorchConnectors!
        #initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
        #initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
        #(b - a) * random_sample() + a

        #TODO Maybe estimator instead of sampler
        qc = Vx(embed_dim)()
        self._vx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:], weight_params=qc.parameters[:embed_dim], input_gradients=True))#, sampler = self.sampler

        qc = xWx(embed_dim)()
        self._xwx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:]+qc.parameters[:embed_dim-1], weight_params=qc.parameters[embed_dim -1:-embed_dim+1], input_gradients=True))#, sampler = self.sampler

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        #TOCHECK maybe leave the plus one out of num_patches for readibility
        #num_patches is including the +1 for the cls token
        #x----> [batch_size, num_patches, emb_dim]
        inp_x = self.layer_norm_1(x)/2

        #vx ----> [batch_size, embed_dim, num_patches]
        vx = torch.empty((inp_x.shape[0], inp_x.shape[2], inp_x.shape[1])).to(device)
        #xwx ---> [batch_size, num_patches, num_patches]
        xwx = torch.empty((inp_x.shape[0], inp_x.shape[1], inp_x.shape[1])).to(device)

        #TOCHECK parameters ---> [batch_size, num_patches, embed_dim - 1]
        #I have to verify that the number pf parameters is coehernt with the paper even if I change the circuits(vector_loader and network)
        #TOCHECK check that each vector in inp_x is at norm 1!!!!
        #FIXME
        parameters = get_RBS_parameters(inp_x)
        #print("parameters", parameters)
        #print("parameters.shape", parameters.shape)

        #TOCHECK The dimension of the embedding determines the number of qubits used in the circuits. That is beacuse
        #naturally it determines the number of components of the vector to be load into the circuit

        #I create this list to obtain the indices for all the states in which we have a qubit in 1, and the others in 0!!!
        #e.g. for embed_dim == num_qubits == 4 : 1:|0001>, 2:|0010>, 4:|0100>, 8:|1000>
        ei = [ 2**j for j in range(0,self.embed_dim)]
        for i in range(self.num_patches):
          vx[:, :, i] = self._vx(parameters[:,i,:])[:, ei]
          #vx[:, :, i] = self._vx(parameters[:,i,:])[:, ei]
          #print("vx[:, :, i]:", vx[:, :, i])

        #I create this list to obtain the indices for all the states in which we have qubit0 in 1.
        #Then we sum the probabilities of each of this states to obtain the probability that this qubit is in 1
        #TOCHECK maybe I have to elevate the sum by 2 to measure the overall probability instead of having
        #only the sum of the amplitudes
        #e.g. for embed_dim == num_qubits == 4 : 1:|0001>, 2:|0010>, 4:|0100>, 8:|1000> 
        ei = [j for j in range(1,2**self.embed_dim, 2)]
        for i in range(self.num_patches):
          for j in range(self.num_patches):
            xwx[:, i,j] = self._xwx(torch.cat((parameters[:,i,:], parameters[:, j, :]), dim = 1))[:,ei]
            #xwx[:, i,j] = torch.sum(self._xwx(torch.cat((parameters[:,i,:], parameters[:, j, :]), dim = 1))[:,ei])
            #print("xwx[:, i,j]:", xwx[:, i,j])

        vx = torch.sqrt(vx) #the output of the circuit is |Vx|^2
        xwx = torch.sqrt(xwx) #the output of the circuit is |xWx|^2
        
        #print("vx:", vx)
        #print("xwx:", xwx)
        #print("vx:", vx.shape)
        #print("xwx:", xwx.shape)
        attn = F.softmax(xwx, dim=-1)
        t = torch.matmul(attn, vx.transpose(1,2))
        #print("t.shape: ", t.shape)
        #print("x.shape: ", x.shape)
        x = x + t
        x = x + self.linear(self.layer_norm_2(x))
        return x
    


class QuantumVisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)

        self.transformer = nn.Sequential(*[quantumAttentionBlock(embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        #print("pre patching", x.shape)
        x = ut.img_to_patch(x, self.patch_size)
        #print("post patching", x.shape)
        B, T, _ = x.shape
        x = self.input_layer(x)
        #print("post self.input_layer", x.shape)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        #x = x.transpose(0, 1)
        x = self.transformer(x)
        #print("self.transformer output: ", x)
        # Perform classification prediction
        cls = x[:, 0, :]
        out = self.mlp_head(cls)
        #print(out.shape)
        return out