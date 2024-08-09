import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.linalg import expm
import functools as ft

#Gates
class Gate:
    """
    Single qubit gate class
    """
    def __init__(self, num_qubits = 1):
        self.num_qubits = num_qubits

        self.I = np.eye(2)
        self.H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])

    def P(self, phi):
        #Phase gate
        return np.exp(1j*phi)*np.eye(2)
        
    def Rx(self, theta): 
        return expm(-1j*X*theta/2)
        
    def Rz(self, theta):
        return expm(-1j*Z*theta/2)

    def Ry(self, theta):
        return expm(-1j*Y*theta/2)

alphabet = "abcdefghijklmnopqrstuvwxyz"

def Multi_gate(gate, k : int, n : int, method = 'kron'):
        """
        gate: ndarray (2,2)
        k: gate qubit index
        n: total number of qubits
        method: how to compute the multi-qubit tensor product
        """
        lst = [np.eye(2) if j != k else gate for j in range(n)]
        #Make the einstring for the tensor product of n tensors
        if method == 'kron':
            return ft.reduce(np.kron, lst)
        elif method == 'einsum':
            
            t = iter(alphabet[0:int(2*n)])
            einstring = ','.join(a+b for a,b in zip(t, t))
            x = alphabet[0:2*n][0::2]
            y = alphabet[0:2*n][1::2]
            return np.einsum(einstring+'->'+x+y,*lst) 
            #So that output has same index convension as ft.reduct(np.kron, list).reshape((2,2,...,2,2,...))
    
def Multi_CNOT(k1 : int, k2 : int, n : int, method ='kron'):
    """
    k1: control qubit index
    k2: target qubit index
    n: total number of qubits
    method: how to compute the multi-qubit tensor product
    """
    cnot0 = np.array([[1,0],[0,0]])
    cnot1 = np.array([[0,0],[0,1]])
    cnotx = np.array([[0,1],[1,0]])
    
    lst1 = [np.eye(2) if j != k1 else cnot0  for j in range(n)]
    lst2 = [cnot1 if j == k1 else cnotx if j == k2 else np.eye(2) for j in range(n) ]

    if method == 'kron':
        return ft.reduce(np.kron,lst1) + ft.reduce(np.kron,lst2) #Optimise this 

    elif method == 'einsum':
    
        #Make the einstring for the tensor product of n tensors
        t = iter(alphabet[0:int(2*n)])
        einstring = ','.join(a+b for a,b in zip(t, t))
        x = alphabet[0:2*n][0::2]
        y = alphabet[0:2*n][1::2]
        return np.einsum(einstring+'->'+x+y,*lst1) +np.einsum(einstring+'->'+x+y,*lst2) 
        #So that output has same index convension as ft.reduct(np.kron, list).reshape((2,2,...,2,2,...))


#State class 
class State:
    def __init__(self, num_qubits :int, init_state = 0):
        """
        init_state: index of basis vector of the initial state, see State().basis for convention
        """
        self.num_qubits = num_qubits
        self.basis = list(it.product(range(2), repeat=num_qubits))
        
        #Initialise the state in init_state
        self.tensor = np.zeros([2 for i in range(num_qubits)],dtype=complex)
        self.tdim = self.tensor.shape
        self.tdim_flat = 2**self.num_qubits
        self.tensor[self.basis[init_state]] = 1

    def apply_single_qubit_gate(self, gate, k :int, method = 'kron'):
        """Applies single qubit gate on multiqubit state
        """
        
        multigate = Multi_gate(gate, k, self.num_qubits, method)
            
        if method == 'einsum':
            multigate = multigate.reshape((self.tdim_flat, self.tdim_flat))
        
        self.tensor = (multigate @ self.tensor.reshape(self.tdim_flat)).reshape(self.tdim)

    def apply_cnot_qubit_gate(self, k1 : int, k2 : int, method = 'kron'):
        """Applies the cnot gate with control qubit k1 and target k2
        """
        multigate = Multi_CNOT(k1,k2, self.num_qubits, method)
        if method =='einsum':
            
            multigate = multigate.reshape((self.tdim_flat, self.tdim_flat))

        self.tensor = (multigate @ self.tensor.reshape(self.tdim_flat)).reshape(self.tdim)


    def conditional_measurement(self, k : int, proj: int):
        """
        k: qubit being measured
        proj: 0 or 1, result of measurement (in computational basis only), 
                        rotate the qubit with H prior to measurement for conjugate basis measurement  
        
        """
        #Swap indices so k is the last qubit
        swapped = self.tensor.swapaxes(k, -1)
        self.tensor = swapped[..., proj] #Project last qubit into 0 or 1 state 

        #Calculate the probability
        norm  = np.sum(self.tensor @ np.conjugate(self.tensor).T)
        if norm != 0:
            self.tensor /= np.sqrt(norm) #Normalise the tensor
        else:
            print(f'Probability of qubit {k} being {proj} is {norm}.')

        #Update properties of reduced state
        self.num_qubits = len(self.tensor)
        self.basis = list(it.product(range(2), repeat=self.num_qubits))
        self.tdim = self.tensor.shape
        self.tdim_flat = 2**self.num_qubits

    def plot_state(self):
        """
        Plot the probability of each basis vector
        """
        probs = np.abs(self.tensor)**2
                    
        plt.bar(range(self.tdim_flat), probs.reshape(self.tdim_flat))
        labels = self.basis

        
        if self.num_qubits <= 4:
            plt.xticks(np.arange(2**self.num_qubits),self.basis, rotation = 90)
            plt.xlabel('Basis state')
        else:
            plt.xlabel('Basis number')
            
        plt.ylabel('Probability')
    

        
       