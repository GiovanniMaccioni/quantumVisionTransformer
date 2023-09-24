from circuits import *

from qiskit_machine_learning.neural_networks import SamplerQNN
import matplotlib.pyplot as plt
from qiskit import transpile
#from qiskit_aer.primitives import Sampler as AerSampler

from qiskit.providers.fake_provider import FakeAuckland
from qiskit import Aer

from qiskit.visualization import plot_gate_map




qc = Vx(8, "theta", "sigma")

qnn = SamplerQNN(
    circuit=qc._circuit,
    input_params=qc.vec_loader.parameters,
    weight_params=qc.V.parameters
)

#Here I can take the backend to simulate one of the ibm machines, and passing it to the 
#AerSimulator to simulate it.
"""backend = FakeAuckland()
plot_gate_map(backend)"""

#TOCHECK if I pass it like this, I get an ideal simulator. The transpile doesn't do anything to the circuit
#I don't know how the simulation happens here
backend = Aer.get_backend('aer_simulator')
#plot_gate_map(backend)


#print(qnn.forward(input_data=[1, 2, 3, 4, 5, 6, 7], weights=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
fig = qnn.circuit.decompose().draw("mpl", scale = 0.3)
plt.show()

fig = qnn.circuit.decompose().decompose().draw("mpl", scale = 0.3)
plt.show()

trans_qc = transpile(qnn.circuit.decompose(), backend)

fig = trans_qc.draw("mpl", scale = 0.3)
plt.show()

print('Original depth:', qnn.circuit.decompose().depth(), 'Decomposed Depth:', trans_qc.depth())

fig = qnn.circuit.decompose().decompose().decompose().draw("mpl", scale = 0.3)
plt.show()

print('Decompose^2:', qnn.circuit.decompose().decompose().depth(), 'Decomposed^3:', qnn.circuit.decompose().decompose().decompose().depth())
