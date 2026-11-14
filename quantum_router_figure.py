"""Load, display, and copy an existing quantum-router image.

Note: This script does not simulate a quantum circuit or create the
underlying figure from experimental/simulation data.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
input_file = PROJECT_DIR / "final_quantum_router.png"
output_file = PROJECT_DIR / "Quantum_Router_Final.png"

if not input_file.exists():
    raise FileNotFoundError(
        f"Figure not found: {input_file}\n"
        "Copy final_quantum_router.png into this script's folder."
    )

print("File found successfully!")
print("Location:", input_file)

with Image.open(input_file) as img:
    print("\nFigure loaded successfully!")
    print("Original size:", img.size)
    print("Image format:", img.format)

    plt.figure(figsize=(17.83, 8.82), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

shutil.copy2(input_file, output_file)

with Image.open(output_file) as result:
    print("\n==========================================")
    print("FINAL FIGURE SAVED SUCCESSFULLY")
    print("==========================================")
    print("Saved as:", output_file)
    print("Final size:", result.size)
