import torch
import numpy as np
import matplotlib.pyplot as plt

from qiskit.visualization import plot_histogram
from qiskit import transpile
#from qiskit import Aer


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.flatten(1,2)
    if flatten_channels:
        x = x.flatten(2,4)
    return x