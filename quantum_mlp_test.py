import torch
import torch.nn as nn
import qiskit as qis

from circuits import *

import data as d

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

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


class HybridMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 256)
        self.linear2 = nn.Linear(256, 4)
        self.linear3 = nn.Linear(4, 10)

        qc = Vx(4, "theta", "sigma")
        self.vx = TorchConnector(SamplerQNN(circuit=qc._circuit, input_params=qc.vec_loader.parameters, weight_params=qc.V.parameters, input_gradients=True))#, sampler = self.sampler

        self.actv = nn.ReLU()
        #self.norm = nn.LayerNorm()
        #self.quantum
        self.ei = [ 2**j for j in range(0,4)]

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)

        x = get_RBS_parameters(x[:,None, :])
        x = torch.sqrt(self.vx(x[:,0,:])[:, self.ei])
        x = self.actv(x)

        x = self.linear3(x)
        return x
    
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
h_mlp = HybridMLP().to(device)

learning_rate=3e-3
epochs = 5


optimizer = torch.optim.Adam(h_mlp.parameters(), lr=learning_rate, eps=1e-4)

train_loader, test_loader = d.get_loaders()

size = len(train_loader.dataset)

h_mlp.train()

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    #training
    running_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        # Compute prediction and loss
        #print(batch)
        X = X.to(device)
        y = y.to(device)

        pred = h_mlp(X)
        #print("pred: ", pred)
        loss = loss_fn(pred, y)#compute the loss 

        # Backpropagation
        optimizer.zero_grad()#zero the gradients for the optimizer
        loss.backward()#backwardpass
        #print(optimizer.param_groups)
        optimizer.step()#optimizer step
        #print the loss once in a while
        """if batch % 1 == 0:
            loss_pok, current = loss.item(), batch * len(X)
            print(f"loss: {loss_pok:>7f}  [{current:>5d}/{size:>5d}]")"""
        running_loss += loss.item()
        print("fin\n")
    
    print(f"Loss Epoch {epoch}: {running_loss/len(train_loader)}")


