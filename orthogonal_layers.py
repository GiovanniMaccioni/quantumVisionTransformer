import qiskit as qis
import torch
import numpy as np

from gates import RBS_gate

class Butterfly: 
  def __init__(self, num_features):
      # --- Circuit definition ---
      self.n_qubits = num_features
      self._circuit = qis.QuantumCircuit(self.n_qubits)
      self.thetas = [] 

      #Number of parameters n_qubits/2 * log(n_qubits)
      self.number_of_parameters = (self.n_qubits//2 * np.log2(self.n_qubits)).astype(int)
      temp = self.n_qubits

      id_gate = 0
      parameters = qis.circuit.ParameterVector("sigma", length=self.number_of_parameters)

      #Building of the circuit
      while (self.n_qubits//temp) != self.n_qubits:

        for i in range(self.n_qubits//2):
          rbs = RBS_gate(parameters[id_gate])()
          id_gate += 1
          step = temp // 2
          idx = i + (i//step)*step
          self._circuit.append(rbs, [idx, idx+step])
          
        self._circuit.barrier()
        temp = temp // 2

  def __call__(self):
    return self._circuit, self.number_of_parameters
    
class Pyramid:
  def __init__(self, num_features):
    # --- Circuit definition ---
    self.n_qubits = num_features
    self._circuit = qis.QuantumCircuit(self.n_qubits)
    self.thetas = [] 

    #Number of parameters n_qubits*(n_qubits-1)/2
    self.number_of_parameters = self.n_qubits*(self.n_qubits - 1)//2


    temp = self.n_qubits
    id_gate = 0
    parameters = qis.circuit.ParameterVector("sigma", length=self.number_of_parameters)

    #Building of the circuit
    for i in range(self.n_qubits-1):
      rbs = RBS_gate(parameters[id_gate])()
      id_gate += 1
      self._circuit.append(rbs, [i, i+1])
      for j in reversed(range(i)): 
        rbs = RBS_gate(parameters[id_gate])()
        id_gate += 1
        self._circuit.append(rbs, [j, j+1])
        
    self._circuit.barrier()
    temp = temp // 2

  def __call__(self):
    return self._circuit, self.number_of_parameters