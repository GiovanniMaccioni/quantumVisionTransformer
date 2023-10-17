import torch
import numpy as np
import matplotlib.pyplot as plt

from qiskit.tools.visualization import plot_histogram
from qiskit import transpile
from qiskit import Aer


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

def temporary():
    shots = 100000
    vec_loader = None

    fig = vec_loader.draw("mpl", scale = 0.5)
    plt.show()

    fig = vec_loader.decompose().draw("mpl", scale = 0.5)
    plt.show()

    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(vec_loader, simulator)#<----

    # Run and get counts
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    print(counts)
    components_values = dict((k, np.sqrt(v/shots)) for k, v in counts.items())
    print(components_values)

    fig = plot_histogram(counts, title='Bell-State counts')
    plt.show()

    return