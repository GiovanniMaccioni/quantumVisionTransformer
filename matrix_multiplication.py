import qiskit as qis
import torch
import numpy as np

from gates import RBS_gate

class Butterfly:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, num_features):
        # --- Circuit definition ---
        self.n_qubits = num_features
        self._circuit = qis.QuantumCircuit(self.n_qubits)
        self.thetas = [] 

        self.number_of_parameters = (self.n_qubits//2 * np.log2(self.n_qubits)).astype(int)
        temp = self.n_qubits
        #print((self.n_qubits//temp) != self.n_qubits)
        id_gate = 0

        parameters = qis.circuit.ParameterVector("sigma", length=self.number_of_parameters)
  
        while (self.n_qubits//temp) != self.n_qubits:
          #print(temp)
          for i in range(self.n_qubits//2):
            #print(id_gate)
            rbs = RBS_gate(parameters[id_gate])()
            id_gate += 1
            #index = self.n_qubits//temp
            step = temp // 2
            idx = i + (i//step)*step
            #print("step:", step)
            #print("idx:", idx, " to idx+1:", idx+step)
            self._circuit.append(rbs, [idx, idx+step])
            
          self._circuit.barrier()
          temp = temp // 2

    def __call__(self):
      return self._circuit