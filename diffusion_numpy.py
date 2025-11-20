import numpy as np
import time
import os
import csv
from PIL import Image

# --- Parametry ---
sizes = [500, 1000, 2000]
steps = 500
rate = 0.3
brightness = 5.0
display_interval = 50
output_base_dir = "frames_numpy"
results_file = "results_numpy.csv"

os.makedirs(output_base_dir, exist_ok=True)

with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["size", "time_s"])

    for size in sizes:
        grid = np.zeros((size, size, 3), dtype=np.float32)
        grid[size//2, size//2] = [1,0,0]
        grid[size//4, size//4] = [0,1,0]
        grid[3*size//4, 3*size//4] = [0,0,1]

        def diffuse(g):
            lap = (g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:] - 4*g[1:-1,1:-1])
            new = g.copy()
            new[1:-1,1:-1] = g[1:-1,1:-1] + rate * lap
            return np.clip(new, 0, 1)

        output_dir = os.path.join(output_base_dir, f"{size}")
        os.makedirs(output_dir, exist_ok=True)

        start = time.time()
        for step in range(steps):
            grid = diffuse(grid)
            if step % display_interval == 0:
                frame = np.clip(grid*brightness, 0, 1)
                Image.fromarray((frame*255).astype(np.uint8)).save(
                    f"{output_dir}/frame_{step:04d}.png"
                )
        end = time.time()
        elapsed = end - start
        print(f"NumPy size {size}: {elapsed:.3f}s")
        writer.writerow([size, elapsed])
