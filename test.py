import torch
import numpy as np

ident = torch.eye(16)
matrix = torch.tensor([[1, 0, 0, 0],
        [0, np.cos(100), np.sin(100), 0],
        [0, - np.sin(100), np.cos(100), 0],
        [0, 0, 0, 1]])

x_gate = torch.clone(ident)
x_gate[1, 0] = 1
x_gate[1, 1] = 0
x_gate[0, 0] = 0
x_gate[0, 1] = 1

matrix = torch.tensor([[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]])

first_matrix = torch.clone(ident)
first_matrix[0:4, 0:4] = matrix

matrix = torch.tensor([[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]])

second_matrix = torch.clone(ident)
second_matrix[0:8:2,0:8:2] = matrix

matrix = torch.tensor([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

third_matrix = torch.clone(ident)
third_matrix[0:16:4,0:16:4] = matrix

result = torch.matmul(x_gate, torch.tensor([1., 0.,0.,0., 0., 0.,0.,0., 0., 0.,0.,0., 0., 0.,0.,0.])) #X_on_q0 matmul |0000>
result = torch.matmul(first_matrix, result)
result = torch.matmul(second_matrix, result)
result = torch.matmul(third_matrix, result)

print("ciao")

"""print(x_gate)
print(first_matrix)
print(second_matrix)
print(third_matrix)"""
print(result)

