import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent

with (BASE / "quirk_ideal_audit_204.csv").open() as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 204

x = np.array([float(r["target_p0"]) for r in rows])
lower = np.array([float(r["unconditional_lower"]) for r in rows])
upper = np.array([float(r["unconditional_upper"]) for r in rows])
theory = np.array([float(r["analytical"]) for r in rows])

det = np.array([not r["state"].startswith("rand_") for r in rows])

error = max(
    np.max(np.abs(lower - theory)),
    np.max(np.abs(upper - theory))
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stix"
})

fig, ax = plt.subplots(figsize=(3.5, 2.8))

xx = np.linspace(0, 1, 500)

ax.plot(xx, (1 + xx) / 2,
        color="black", linewidth=1.2,
        label="Analytical prediction")

ax.scatter(x[~det], lower[~det],
           s=12, facecolors="none",
           edgecolors="#2166AC",
           linewidths=0.7,
           label="Qiskit (both receivers)")

ax.scatter(x[det], lower[det],
           marker="D", s=28,
           color="#B2182B",
           label="Deterministic states",
           zorder=5)

ax.axhline(0.75, color="gray",
           linestyle="--", linewidth=0.9,
           label="Haar mean (0.7500)")

ax.set_xlabel(r"$|\langle 0|\phi\rangle|^2$")
ax.set_ylabel("Unconditional receiver fidelity")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.48, 1.02)

ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0.5, 1, 6))

ax.grid(alpha=0.18)
ax.legend(loc="lower right", frameon=True)

fig.tight_layout()

fig.savefig(BASE / "fig9_routing_fidelity.png",
            dpi=600, bbox_inches="tight")

fig.savefig(BASE / "fig9_routing_fidelity.svg",
            bbox_inches="tight")

plt.close(fig)

print("Input states:", len(rows))
print("Lower receiver mean:", f"{lower.mean():.8f}")
print("Upper receiver mean:", f"{upper.mean():.8f}")
print("Analytical mean:", f"{theory.mean():.8f}")
print("Maximum deviation:", f"{error:.3e}")
print("Figure exported successfully: PNG and SVG")
