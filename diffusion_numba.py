import numpy as np
import time
import os
import csv
from PIL import Image
from typing import List

# --- KONFIGURACJA ---
SIZES: List[int] = [500, 1000, 2000]
STEPS: int = 1000
RATE: float = 0.3
BRIGHTNESS: float = 5.0
DISPLAY_INTERVAL: int = 50
OUTPUT_BASE_DIR: str = "frames_numpy"
RESULTS_FILE: str = "results_numpy.csv"


def update_grid(inp: np.ndarray, out: np.ndarray) -> None:
    """
    Oblicza jeden krok dyfuzji przy użyciu standardowych operacji wektorowych NumPy.

    Args:
        inp: Siatka wejściowa (stan obecny).
        out: Bufor na siatkę wyjściową (stan nowy).
    """
    # Obliczenie Laplasjanu na wycinkach (slicing).
    # Operacje te tworzą tymczasowe tablice w pamięci, co jest narzutem NumPy.
    lap = (inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:] - 4 * inp[1:-1, 1:-1])

    # Kopiujemy wnętrze, aby zachować brzegi (warunki brzegowe = 0)
    # W przeciwieństwie do wersji inp.copy(), tutaj używamy prealokowanego bufora out.
    out[1:-1, 1:-1] = inp[1:-1, 1:-1] + RATE * lap

    # Ograniczenie wartości do zakresu [0, 1] (operacja in-place)
    np.clip(out, 0, 1, out=out)


def save_frame(grid: np.ndarray, step: int, output_dir: str) -> None:
    """Zapisuje aktualny stan siatki do pliku PNG."""
    frame = np.clip(grid * BRIGHTNESS, 0, 1)
    img = Image.fromarray((frame * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{step:04d}.png"))


def run_simulation(size: int, writer: csv.writer) -> None:
    """
    Przeprowadza symulację dla jednego rozmiaru siatki (Single Threaded NumPy).
    """
    output_dir = os.path.join(OUTPUT_BASE_DIR, str(size))
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- NumPy (Baseline) size: {size} ---")

    # Inicjalizacja siatki
    grid = np.zeros((size, size, 3), dtype=np.float32)
    # Prealokacja bufora wyjściowego (unikamy alokacji w pętli)
    new_grid = np.zeros_like(grid)

    # Warunki początkowe
    grid[size // 2, size // 2] = [1, 0, 0]
    grid[size // 4, size // 4] = [0, 1, 0]
    grid[3 * size // 4, 3 * size // 4] = [0, 0, 1]

    start_time = time.time()

    for step in range(STEPS):
        update_grid(grid, new_grid)

        # Zamiana buforów (wskaźników), zamiast tworzenia nowych obiektów
        grid, new_grid = new_grid, grid

        if step % DISPLAY_INTERVAL == 0:
            save_frame(grid, step, output_dir)

    elapsed = time.time() - start_time
    print(f"Czas wykonania: {elapsed:.3f}s")

    writer.writerow([size, elapsed])


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["size", "time_s"])

        for size in SIZES:
            run_simulation(size, writer)


if __name__ == "__main__":
    main()