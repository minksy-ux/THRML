import pennylane as qml
from pennylane import numpy as np

n_qubits = 10
skip = 10
dev = qml.device('default.qubit', wires=n_qubits, shots=10000)

@qml.qnode(dev, interface="jax")
def crec_hqge_qnn(params=None):
    # Start with |+> for all qubits (superposition)
    for q in range(n_qubits):
        qml.Hadamard(q)
    # Entangle each pair by 10-skip-gap
    for q in range(n_qubits):
        qml.CZ(q, (q + skip) % n_qubits)
    # (Optional) If params given, do bias/B-field rotations
    if params is not None:
        for i in range(n_qubits):
            qml.RZ(params[i], wires=i)
    # Return measurement in Z-basis
    return qml.sample(qml.PauliZ(wires=list(range(n_qubits))))

# Example: Sample bit-strings from the HQGE quantum circuit
samples = crec_hqge_qnn()
bit_samples = (1 - samples) // 2   # map ±1 -> {0,1}
print(bit_samples[:10])