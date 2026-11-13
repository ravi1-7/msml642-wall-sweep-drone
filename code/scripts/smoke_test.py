"""Hover for 5 s and print how well height was held. Run: uv run python scripts/smoke_test.py"""

import numpy as np

from wallsweep.sim import hover

trace = hover(seconds=5.0, height=1.0)
z = trace[-48:, 2]
print(f"final position: {np.round(trace[-1], 3)}")
print(f"last second height: mean {z.mean():.3f} m, max error {np.abs(z - 1).max() * 100:.1f} cm")
