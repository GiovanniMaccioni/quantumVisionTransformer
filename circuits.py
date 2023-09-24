import qiskit as qis
import numpy as np

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
        #print(thetas)
        index = 0
        for param in self._circuit.parameters:
          #print(len(self._circuit.parameters))
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
    
class Vx:#DEBUG Added here hadamard gates instead of keeping them in the RBS_Gate class
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, vector_parameter_name, V_parameter_name):
        # --- Circuit definition ---
        self.num_qubits = n_qubits ## Aggiunto per compatibilità con EstimatorQNN
        self._circuit = qis.QuantumCircuit(n_qubits)
        self.vec_loader = VectorLoader(n_qubits, vector_parameter_name)()
        self.V = Butterfly(n_qubits, V_parameter_name)()
        #self._circuit.h([0,1, 2, 3])#DEBUG
        self._circuit.compose(self.vec_loader.compose(self.V), inplace = True) 
        #self._circuit.h([0,1, 2, 3])#DEBUG
        #TODO take the z expectaion value _circuit.exp_val<-----
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
        #TODO take the z expectaion value _circuit.exp_val<-----   
        self._circuit.measure_all()
        # -------------------------