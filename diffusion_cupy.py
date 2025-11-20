import numpy as np
import cupy as cp
import time
import os
import csv
from PIL import Image
from typing import List, Tuple

# --- KONFIGURACJA ---
SIZES: List[int] = [500, 1000, 2000]
STEPS: int = 1000
RATE: float = 0.3
BRIGHTNESS: float = 5.0
DISPLAY_INTERVAL: int = 50
OUTPUT_BASE_DIR: str = "frames_cupy"
RESULTS_FILE: str = "results_cupy.csv"


def update_grid(inp: cp.ndarray, outp: cp.ndarray) -> None:
    """
    Oblicza jeden krok symulacji dyfuzji przy użyciu Laplasjanu na GPU.
    Operacje wykonywane są wektorowo (slicing) bez pętli for.

    Args:
        inp: Siatka wejściowa (stan obecny) na GPU.
        outp: Bufor na siatkę wyjściową (stan nowy) na GPU.
    """
    # Obliczenie Laplasjanu (sąsiedzi góra, dół, lewo, prawo - środek)
    lap = (inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:] - 4 * inp[1:-1, 1:-1])

    # Aktualizacja stanu i przycięcie wartości do zakresu [0, 1]
    outp[1:-1, 1:-1] = inp[1:-1, 1:-1] + RATE * lap
    cp.clip(outp, 0, 1, out=outp)


def save_frame(grid_gpu: cp.ndarray, step: int, output_dir: str) -> None:
    """
    Pobiera dane z GPU do pamięci RAM i zapisuje klatkę jako obraz PNG.
    Operacja cp.asnumpy() jest kosztowna, więc nie należy jej nadużywać w pętli pomiarowej.
    """
    # Transfer VRAM (GPU) -> RAM (CPU)
    frame_cpu = cp.asnumpy(grid_gpu)

    # Normalizacja i zapis
    frame_disp = np.clip(frame_cpu * BRIGHTNESS, 0, 1)
    img = Image.fromarray((frame_disp * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{step:04d}.png"))


def run_simulation(size: int, writer: csv.writer) -> None:
    """
    Przygotowuje środowisko i przeprowadza symulację dla zadanego rozmiaru siatki.
    Mierzy czas wykonania z uwzględnieniem synchronizacji CUDA.
    """
    output_dir = os.path.join(OUTPUT_BASE_DIR, str(size))
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- CuPy (GPU) size: {size} ---")

    # Inicjalizacja danych bezpośrednio na GPU
    grid = cp.zeros((size, size, 3), dtype=cp.float32)
    new_grid = cp.zeros_like(grid)

    # Ustawienie warunków początkowych
    grid[size // 2, size // 2] = cp.array([1, 0, 0], dtype=cp.float32)
    grid[size // 4, size // 4] = cp.array([0, 1, 0], dtype=cp.float32)
    grid[3 * size // 4, 3 * size // 4] = cp.array([0, 0, 1], dtype=cp.float32)

    # Ważne: Synchronizacja strumienia CUDA przed startem zegara
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()

    for step in range(STEPS):
        update_grid(grid, new_grid)
        grid, new_grid = new_grid, grid  # Swap referencji (wskaźników)

        if step % DISPLAY_INTERVAL == 0:
            save_frame(grid, step, output_dir)

    # Ważne: Synchronizacja strumienia CUDA przed zatrzymaniem zegara,
    # aby upewnić się, że GPU zakończyło wszystkie zlecone zadania.
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time

    print(f"Czas wykonania: {elapsed:.3f}s")
    writer.writerow([size, elapsed])


def main() -> None:
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["size", "time_s"])

        for size in SIZES:
            run_simulation(size, writer)


if __name__ == "__main__":
    main()