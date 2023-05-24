import numpy as np
import qiskit

class VirtualQuantumBard:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = qiskit.QuantumCircuit(num_qubits)

    def add_gate(self, gate, qubits):
        self.circuit.append(gate, qubits)

    def measure(self, qubits):
        self.circuit.measure(qubits)

    def execute(self):
        result = qiskit.execute(self.circuit, backend=qiskit.Aer.get_backend('qasm_simulator')).result()
        return result.get_counts()

    def get_probability(self, state):
        counts = self.execute()
        return counts[state] / len(counts)

