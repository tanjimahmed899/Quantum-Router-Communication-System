"""Isolated three-qubit transfer: illustrative logical-gate depolarizing noise.
Requires verify_quirk_qiskit.py and quirk_ideal_audit_204.csv in this folder.
Run: python verify_quirk_noise_qiskit.py
These are NEW assumed parameters, not a recovery of the original noise model.
"""
import csv
import json
from pathlib import Path
import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error
from verify_quirk_qiskit import test_states, transfer

OUT = Path(__file__).resolve().parent
# lambda in D_lambda(rho)=(1-lambda)rho+lambda I/d, on gate operands.
# Logical gate model (including QFT3 as a single 3-qubit gate); NOT hardware-native.
MODELS = {'ideal': {1: 0., 2: 0., 3: 0.},
          'low_noise': {1: .005, 2: .01, 3: .02},
          'higher_noise': {1: .02, 2: .04, 3: .08}}


def read_states():
    with (OUT / 'quirk_ideal_audit_204.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    generated = list(test_states())
    if len(rows) != 204 or len(generated) != 204:
        raise ValueError('Expected exactly 204 states.')
    states = []
    for row, (label, expected) in zip(rows, generated):
        v = np.array([complex(float(row['alpha_real']), float(row['alpha_imag'])),
                      complex(float(row['beta_real']), float(row['beta_imag']))])
        if row['state'] != label or not np.allclose(v, expected, atol=1e-12, rtol=0):
            raise ValueError(f'Dataset mismatch: {label}')
        if not np.isclose(np.linalg.norm(v), 1., atol=1e-12, rtol=0):
            raise ValueError(f'Non-normalized state: {label}')
        states.append(v / np.linalg.norm(v))
    return rows, states


def build_circuit(v, logical, channels):
    qc = QuantumCircuit(3)
    # Perfect input preparation, ancillas q1 and q2 start at |0>.
    full = np.zeros(8, dtype=complex)
    full[:2] = v  # Qiskit little endian: input q0
    qc.set_density_matrix(np.outer(full, full.conj()))
    for instruction in logical.data:
        wires = [logical.find_bit(q).index for q in instruction.qubits]
        qc.append(instruction.operation, wires)
        if len(wires) in channels:
            qc.append(channels[len(wires)], wires)
    qc.save_density_matrix()
    return qc


def main():
    rows, states = read_states()
    logical = QuantumCircuit(3)
    transfer(logical, 0, 1, 2)
    sim = AerSimulator(method='density_matrix')
    summary = {}
    for name, rates in MODELS.items():
        # Explicit exact CPTP channels, not stochastic shots or calibration data.
        channels = {k: Kraus(depolarizing_error(lam, k).to_quantumchannel()).to_instruction()
                    for k, lam in rates.items() if lam > 0}
        circuits = [build_circuit(v, logical, channels) for v in states]
        # No transpilation: the logical gates and their error boundaries stay intact.
        result = sim.run(circuits, shots=1).result()
        if not result.success:
            raise RuntimeError(str(result.status))
        vals = []
        for i, (v, row) in enumerate(zip(states, rows)):
            rho = DensityMatrix(result.data(i)['density_matrix'])
            receiver = partial_trace(rho, [0, 1])
            fidelity = float(state_fidelity(Statevector(v), receiver))
            if not np.isfinite(fidelity) or not -1e-10 <= fidelity <= 1 + 1e-10:
                raise ValueError(f'Invalid fidelity: {row["state"]}, {name}')
            if name == 'ideal' and abs(fidelity - float(row['isolated'])) > 1e-10:
                raise ValueError(f'Ideal audit mismatch: {row["state"]}')
            vals.append(fidelity)
            row['noise_audit_' + name] = fidelity
        vals = np.array(vals)
        summary[name] = {'deterministic_mean': float(vals[:4].mean()),
                         'random_mean': float(vals[4:].mean()),
                         'all_mean': float(vals.mean()),
                         'min': float(vals.min()), 'max': float(vals.max())}
    print('Qiskit:', qiskit.__version__, '| Aer:', qiskit_aer.__version__, '| NumPy:', np.__version__)
    print('204 states (4 deterministic + 200 Haar-random); seed 2026')
    print('Logical gate counts:', dict(logical.count_ops()))
    print('Depolarizing lambda by gate arity (1q, 2q, 3q):', MODELS)
    print(f'{"Condition":20s} {"Deterministic":>14s} {"Random":>12s} {"All":>12s}')
    for name, s in summary.items():
        print(f'{name:20s} {s["deterministic_mean"]:14.8f} {s["random_mean"]:12.8f} {s["all_mean"]:12.8f}')
    output = OUT / 'quirk_noise_audit_204.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {'input_csv': 'quirk_ideal_audit_204.csv', 'seed': 2026,
                'qiskit_version': qiskit.__version__, 'aer_version': qiskit_aer.__version__,
                'numpy_version': np.__version__, 'logical_gate_counts': dict(logical.count_ops()),
                'depolarizing_lambda': MODELS, 'model': 'logical gate operands; ideal preparation; no measurement errors; no hardware calibration',
                'results': summary}
    metadata_file = OUT / 'quirk_noise_metadata.json'
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('CSV saved to', output)
    print('Metadata saved to', metadata_file)

if __name__ == '__main__':
    main()
