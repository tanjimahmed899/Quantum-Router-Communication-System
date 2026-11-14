# Quantum Router Figure – Display and Export

This repository stores a final quantum-router figure and a Python utility to
load, display, and copy the existing image. **The script does not generate the
router circuit or simulation results.** To reproduce the scientific results,
the original circuit definition, simulator scripts, configuration, and data
must be provided separately.

## Files

- `quantum_router_figure.py`: Loads, displays, and copies the figure.
- `final_quantum_router.png`: Source figure (**add this image from your PC**).
- `Quantum_Router_Final.png`: Copy produced by the script after execution.
- `requirements.txt`: Python dependencies.

## How to run

1. Put `final_quantum_router.png` in the same folder as the script.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Run: `python quantum_router_figure.py`

The script saves a copy named `Quantum_Router_Final.png` in the same folder.

## Reproducibility note

This code is for figure handling only. Upload original Qiskit/Quirk circuit
implementation and input data separately if the paper is intended to support
reproduction of fidelity or router simulation results.
