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

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

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
        print(thetas)
        index = 0
        for param in self._circuit.parameters:
          print(len(self._circuit.parameters))
          self._circuit = self._circuit.bind_parameters({param: thetas[index]})
          index += 1
    
      return self._circuit

class Butterfly:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, num_features, parameter_name):
        # --- Circuit definition ---
        self.n_qubits = num_features
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self.thetas = [] 
        #self.num_gates = self.n_qubits//2 * torch.log2(self.n_qubits)
        #self.num_gates_each_level = self.num_qubits//2
        self.num_gates = self.n_qubits//2 * np.log2(self.n_qubits)
        temp = self.n_qubits
        #print((self.n_qubits//temp) != self.n_qubits)
        id_gate = 0
        
        while (self.n_qubits//temp) != self.n_qubits:
          #print(temp)
          for i in range(self.n_qubits//2):
            parameter = parameter_name + "{:02d}"
            parameter = parameter.format(id_gate)
            #print(id_gate)
            id_gate += 1
            rbs = RBS_gate(parameter)
            #index = self.n_qubits//temp
            step = temp // 2
            idx = i + (i//step)*step
            #print("step:", step)
            #print("idx:", idx, " to idx+1:", idx+step)
            self._circuit.append(rbs(), [idx, idx+step])
            
          self._circuit.barrier()
          temp = temp // 2

    def __call__(self, thetas=None):
      if thetas != None:
        #all_thetas = [i for i in range(len(thetas))]
        print(thetas)
        index = 0
        for param in self._circuit.parameters:
          print(len(self._circuit.parameters))
          self._circuit = self._circuit.bind_parameters({param: thetas[index]})
          index += 1
    
      return self._circuit


class Vx:

    def __init__(self, n_qubits, vector_parameter_name, V_parameter_name):
        # --- Circuit definition ---
        self.num_qubits = n_qubits ## Aggiunto per compatibilità con EstimatorQNN
        self._circuit = qis.QuantumCircuit(n_qubits)
        self.vec_loader = VectorLoader(n_qubits, vector_parameter_name)()
        self.V = Butterfly(n_qubits, V_parameter_name)()
        self._circuit.compose(self.vec_loader.compose(self.V), inplace = True) 
        self._circuit.measure_all()
        self.parameters = self._circuit.parameters
        # ---------------------------


class xWx:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, i_vector_parameter_name, W_parameter_name, j_vector_parameter_name):
        # --- Circuit definition ---
        self.num_qubits = n_qubits ## Aggiunto per compatibilità con EstimatorQNN
        self._circuit = qis.QuantumCircuit(n_qubits)
        self.vec_loader = VectorLoader(n_qubits, i_vector_parameter_name)()
        self.W = Butterfly(n_qubits, W_parameter_name)()
        self.vec_loader_adjoint = VectorLoader(n_qubits, j_vector_parameter_name)().inverse()
        self._circuit.compose(self.vec_loader.compose(self.W.compose(self.vec_loader_adjoint)), inplace = True)   
        self._circuit.measure_all()
        # ---------------------------


class quantumAttentionBlock_PART1(nn.Module):

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

        self.embed_dim = embed_dim

        self.num_patches = num_patches + 1#+1 because it takes into account the class vector
        

        #Missing initialization of weights before TorchConnectors!

        qc = Vx(embed_dim, "theta", "sigma")
        self._vx = TorchConnector(SamplerQNN(circuit=qc._circuit, input_params=list(qc.vec_loader.parameters), weight_params=qc.V.parameters, input_gradients=True))#, sampler = self.sampler

        qc = xWx(embed_dim, "theta", "sigma", "phi")
        self._xwx = TorchConnector(SamplerQNN(circuit=qc._circuit, input_params= list(qc.vec_loader.parameters) + list(qc.vec_loader_adjoint.parameters), weight_params=qc.W.parameters, input_gradients=True))#, sampler = self.sampler


    def forward(self, x):
        inp_x = x

        #No batch accounted for; check if qiskit is compatible with batch
        vx = torch.empty((inp_x.shape[0], inp_x.shape[2], inp_x.shape[1])).to(device)
        xwx = torch.empty((inp_x.shape[0], inp_x.shape[1], inp_x.shape[1])).to(device)

        parameters = ut.get_RBS_parameters(inp_x)
        #print("parameters", parameters)
        #print("parameters.shape", parameters.shape)
        ei = [ 2**j for j in range(0,self.embed_dim) ]
        print(ei)
        for i in range(self.num_patches):
          #vx[:, :, i] = self._vx(parameters[:,i,:])[:, 0:self.embed_dim]
          vx[:, :, i]  = self._vx(parameters[:,i,:])[:, ei]
          #print("vx[:, :, i]:", vx[:, :, i])
          """ 
          b = torch.tensor([[0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001],
              [0.001, 0., 0., 0.001]])
          """
          #max = (torch.where(vx[:,:,i] > 0, vx[:,:,i], 1.0)).min() * 1e-3
          #print("max------>", max)
          #b = max*torch.rand(1)
          #print("b------>", b)
          #vx[:, :, i] += b

        for i in range(self.num_patches):
          for j in range(self.num_patches):
            xwx[:, i,j] = self._xwx(torch.cat((parameters[:,i,:], parameters[:, j, :]), dim = 1))[:,0]
            #print("xwx[:, i,j]:", xwx[:, i,j])

        
        vx = torch.sqrt(vx) #the output of the circuit is |Vx|^2
        xwx = torch.sqrt(xwx) #the output of the circuit is |xWx|^2
        
        #print("vx:", vx)
        #print("xwx:", xwx)
        #print("vx:", vx.shape)
        #print("xwx:", xwx.shape)
        attn = F.softmax(xwx/torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.int8)), dim=-1)
        t = torch.matmul(attn, vx.transpose(1,2))
        #print("t.shape: ", t.shape)
        #print("x.shape: ", x.shape)
        x = x + t
        return x


class quantumAttentionBlock_PART2(nn.Module):

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
        
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
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
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)#FIXME added larìyer_norm1 here

        self.quantum_p1 = quantumAttentionBlock_PART1(embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout)
        self.quantum_p2 = quantumAttentionBlock_PART2(embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout)

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
        x = self.layer_norm_1(x)
        #x = x.transpose(0, 1)
        x = self.quantum_p1(x)
        x = self.quantum_p2(x)
#        print("self.transformer output: ", x)
        # Perform classification prediction
        cls = x[:,0, :]
        out = self.mlp_head(cls)
#        print(out.shape)
        return out