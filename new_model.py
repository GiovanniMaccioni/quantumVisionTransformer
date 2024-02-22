import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit.visualization import *
from qiskit_aer import AerSimulator


from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import Sampler

from qiskit_algorithms.utils import algorithm_globals

import utils as U

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import circuits as C


class quantumAttentionBlock(nn.Module):

    def __init__(self, vec_loader, matrix_mul, embed_dim, hidden_dim, num_heads, num_patches, dropout=0.0):
        """
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim

        self.num_patches = num_patches + 1#+1 because it takes into account the class token vector

        aersim = AerSimulator(method='statevector', device='GPU')
        sampler = Sampler()
        sampler.set_options(backend=aersim)

        self.vx = C.Vx(embed_dim, vec_loader, matrix_mul)
        qc, num_weights = self.vx()
        self._vx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:], weight_params=qc.parameters[:num_weights], input_gradients=True, sampler = sampler))

        qc, num_weights = C.xWx(embed_dim, vec_loader, matrix_mul)()
        self._xwx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:]+qc.parameters[:embed_dim-1], weight_params=qc.parameters[embed_dim -1:-embed_dim+1], input_gradients=True, sampler = sampler))

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #x ----> [batch_size, num_patches, emb_dim]
        inp_x = self.layer_norm_1(x)

        #We need tensors to save the results of the multiplications
        #vx ----> [batch_size, embed_dim, num_patches]
        vx = torch.empty((inp_x.shape[0], inp_x.shape[2], inp_x.shape[1])).to(device)
        #xwx ---> [batch_size, num_patches, num_patches]
        xwx = torch.empty((inp_x.shape[0], inp_x.shape[1], inp_x.shape[1])).to(device)

        #Normailize the input vectors
        inp_x = inp_x/torch.sqrt(torch.sum(torch.pow(inp_x, 2)+1e-4, dim=1, keepdim=True)+1e-8)

        #Compute the embed_dim-1 parameters
        parameters = self.vx.get_RBS_parameters(inp_x)

        #I create this list to obtain the indices for all the states in which we have a qubit in 1, and the others in 0!!!
        #e.g. for embed_dim == num_qubits == 4 : 1:|0001>, 2:|0010>, 4:|0100>, 8:|1000>
        ei = [ 2**j for j in range(0,self.embed_dim)]
        for i in range(self.num_patches):
          vx[:, :, i] = self._vx(parameters[:,i,:])[:, ei]
        

        #I create this list to obtain the indices for all the states in which we have qubit0 in 1.
        #Then we sum the probabilities of each of this states to obtain the probability that this qubit is in 1
        ei = [j for j in range(1,2**self.embed_dim, 2)]
        for i in range(self.num_patches):
          for j in range(self.num_patches):
            p = torch.cat((parameters[:,i,:], parameters[:, j, :]), dim = 1)
            xwx[:, i,j] = torch.sum(self._xwx(p)[:,ei], dim = 1)

        vx = torch.sqrt(vx + 1e-8) #the output of the circuit is |Vx|^2
        xwx = torch.sqrt(xwx + 1e-8) #the output of the circuit is |xWx|^2
        
        #
        attn = F.softmax(xwx, dim=-1)
        t = torch.matmul(attn, vx.transpose(1,2))
        x = x + t

        x = x + self.linear(self.layer_norm_2(x))
        return x
    


class QuantumVisionTransformer(nn.Module):

    def __init__(self,embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0,  vec_loader='diagonal', matrix_mul = 'pyramid'):
        """
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)

        self.transformer = nn.Sequential(*[quantumAttentionBlock(vec_loader, matrix_mul, embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))

    def forward(self, x):
        x = U.img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[:, 0, :]
        out = self.mlp_head(cls)
        return out