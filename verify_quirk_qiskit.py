"""Ideal audit of the supplied Quirk circuit using Qiskit.
Qubit labels in paper q1..q10 map to Python 0..9. The first two idle
Quirk wires are omitted. For arbitrary input tests only, replace the
animated X^t, Y^t, H source preparation by explicit state preparation;
set routing-control preparation independently. All subsequent unitary
operations follow the supplied link. No noise model is assumed.
"""
import csv
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

SEED = 2026
RX = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
# Quirk's QFT3: positive-exponent DFT, first wire least significant.
QFT3 = np.exp(2j * np.pi * np.outer(np.arange(8), np.arange(8)) / 8) / np.sqrt(8)

def test_states():
    yield 'zero', np.array([1, 0], complex)
    yield 'one', np.array([0, 1], complex)
    yield 'plus', np.array([1, 1], complex) / np.sqrt(2)
    yield 'plus_i', np.array([1, 1j], complex) / np.sqrt(2)
    rng = np.random.default_rng(SEED)
    for i in range(200):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        if abs(v[0]) > 0:
            v *= np.exp(-1j * np.angle(v[0]))
        yield f'rand_{i:03d}', v

def transfer(qc, a, b, r):
    qc.h(a); qc.h(b); qc.h(r)
    qc.rz(2 * np.pi, a)
    qc.unitary(QFT3, [a, b, r], label='Quirk QFT3')
    qc.h(r)
    qc.ccx(a, b, r)
    qc.rxx(np.pi/2, a, b)
    qc.h(r)

def isolated(v):
    qc = QuantumCircuit(3)
    qc.prepare_state(v, [0])
    transfer(qc, 0, 1, 2)
    return Statevector.from_instruction(qc)

def integrated(v, control):
    qc = QuantumCircuit(10)
    qc.prepare_state(v, [3])
    if control == 1:
        qc.x(5)
    elif control == '+':
        qc.h(5)
    qc.rx(np.pi/2, 3)
    qc.cswap(5, 3, 4)
    qc.swap(0, 4)
    qc.swap(3, 6)
    qc.cx(6, 7)
    # Quirk <<2 is a two-wire cyclic rotation, equivalent to SWAP.
    qc.swap(6, 7); qc.swap(6, 7)
    qc.cx(7, 6)
    transfer(qc, 0, 1, 2)
    transfer(qc, 7, 8, 9)
    return Statevector.from_instruction(qc)

def receiver_fidelity(s, target, q):
    rho = partial_trace(s, [k for k in range(s.num_qubits) if k != q])
    return float(state_fidelity(Statevector(target), rho))

def conditioned(s, value):
    vec = s.data.copy()
    bits = (np.arange(len(vec)) >> 5) & 1
    vec[bits != value] = 0
    p = float(np.vdot(vec, vec).real)
    return p, Statevector(vec / np.sqrt(p))

def main():
    rows = []
    for name, v in test_states():
        target = RX @ v
        plus = integrated(v, '+')
        p0, c0 = conditioned(plus, 0)
        p1, c1 = conditioned(plus, 1)
        # Control 0 selects the lower receiver q10; control 1 selects q3.
        rows.append(dict(state=name, alpha_real=v[0].real, alpha_imag=v[0].imag,
                         beta_real=v[1].real, beta_imag=v[1].imag,
                         isolated=receiver_fidelity(isolated(v), v, 2),
                         fixed_lower=receiver_fidelity(integrated(v, 0), target, 9),
                         fixed_upper=receiver_fidelity(integrated(v, 1), target, 2),
                         conditional_lower=receiver_fidelity(c0, target, 9),
                         conditional_upper=receiver_fidelity(c1, target, 2),
                         unconditional_lower=receiver_fidelity(plus, target, 9),
                         unconditional_upper=receiver_fidelity(plus, target, 2),
                         p0=p0, p1=p1, target_p0=float(abs(target[0])**2),
                         analytical=float((1+abs(target[0])**2)/2)))
    cols = ['isolated', 'fixed_lower', 'fixed_upper', 'conditional_lower',
            'conditional_upper', 'unconditional_lower', 'unconditional_upper',
            'p0', 'p1', 'analytical']
    print(f'Input states: {len(rows)} (4 deterministic + 200 Haar-random); seed={SEED}')
    print(f'{"Metric":25s} {"Deterministic":>14s} {"Random":>12s} {"All":>12s}')
    for k in cols:
        a = np.array([r[k] for r in rows])
        print(f'{k:25s} {a[:4].mean():14.8f} {a[4:].mean():12.8f} {a.mean():12.8f}')
    err = max(abs(r[k] - r['analytical']) for r in rows
              for k in ['unconditional_lower', 'unconditional_upper'])
    print(f'Maximum absolute difference vs analytical prediction: {err:.3e}')
    out = Path(__file__).with_name('quirk_ideal_audit_204.csv')
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'CSV saved to {out}')

if __name__ == '__main__':
    main()
