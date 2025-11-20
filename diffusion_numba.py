import numpy as np
from numba import njit, prange, set_num_threads
import time
import os
import csv
from PIL import Image
from typing import List

# --- KONFIGURACJA (Dostosowana pod serwer Torus: 32 CPU) ---
SIZES: List[int] = [500, 1000, 2000]
STEPS: int = 500
RATE: float = 0.3
BRIGHTNESS: float = 5.0
DISPLAY_INTERVAL: int = 50
OUTPUT_BASE_DIR: str = "frames_numba"
RESULTS_FILE: str = "results_numba.csv"

THREADS_TO_TEST: List[int] = [1, 8, 16, 32]


@njit(parallel=True, fastmath=True)
def update_grid_numba(inp: np.ndarray, out: np.ndarray) -> None:
    """
    Wykonuje krok dyfuzji oraz przycinanie (clipping) wartości.

    Używamy dekoratora @njit(parallel=True), który automatycznie zrównolegla
    pętle prange na dostępne wątki procesora.
    """
    rows, cols, channels = inp.shape

    # 1. Obliczenia dyfuzji (Laplasjan)
    # prange informuje Numbę, że ta pętla może być bezpiecznie wykonana równolegle
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            for c in range(channels):
                out[i, j, c] = inp[i, j, c] + RATE * (
                        inp[i - 1, j, c] +
                        inp[i + 1, j, c] +
                        inp[i, j - 1, c] +
                        inp[i, j + 1, c] -
                        4 * inp[i, j, c]
                )

    # 2. Clipowanie wartości do zakresu [0, 1]
    for i in prange(rows):
        for j in range(cols):
            for c in range(channels):
                val = out[i, j, c]
                if val < 0:
                    out[i, j, c] = 0
                elif val > 1:
                    out[i, j, c] = 1


def save_frame(grid: np.ndarray, step: int, output_dir: str) -> None:
    """Zapisuje aktualny stan siatki do pliku PNG."""
    frame = np.clip(grid * BRIGHTNESS, 0, 1)
    img = Image.fromarray((frame * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{step:04d}.png"))


def run_simulation(size: int, threads: int, writer: csv.writer, should_save_images: bool) -> None:
    """
    Przeprowadza symulację dla zadanego rozmiaru i liczby wątków.
    """
    # Ustawiamy liczbę wątków dla Numby na tę konkretną iterację
    set_num_threads(threads)

    output_dir = os.path.join(OUTPUT_BASE_DIR, str(size))
    if should_save_images:
        os.makedirs(output_dir, exist_ok=True)

    print(f"--- Numba Processing size: {size} | Threads: {threads} ---")

    # Inicjalizacja siatki
    grid = np.zeros((size, size, 3), dtype=np.float32)
    new_grid = grid.copy()

    # Warunki początkowe
    grid[size // 2, size // 2] = [1, 0, 0]
    grid[size // 4, size // 4] = [0, 1, 0]
    grid[3 * size // 4, 3 * size // 4] = [0, 0, 1]

    # Start pomiaru czasu
    # Pierwsze wywołanie funkcji @njit zawiera narzut kompilacji,
    # ale przy 500-1000 krokach jest on pomijalny w ogólnym rozrachunku.
    start_time = time.time()

    for step in range(STEPS):
        update_grid_numba(grid, new_grid)

        # Swap referencji (szybka zamiana buforów)
        grid, new_grid = new_grid, grid

        if should_save_images and step % DISPLAY_INTERVAL == 0:
            save_frame(grid, step, output_dir)

    elapsed = time.time() - start_time
    print(f"    Zakończono: {elapsed:.3f}s")

    writer.writerow([size, threads, elapsed])


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["size", "threads", "time_s"])

        for size in SIZES:
            # Iterujemy po konfiguracjach wątków
            for i, threads in enumerate(THREADS_TO_TEST):
                # Optymalizacja: Zapisujemy obrazy tylko dla pierwszego testu (i=0).
                # Wynik graficzny jest identyczny niezależnie od liczby wątków.
                save_images_flag = (i == 0)

                run_simulation(size, threads, writer, save_images_flag)


if __name__ == "__main__":
    main()