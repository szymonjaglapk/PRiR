from mpi4py import MPI
import numpy as np
import time
import os
from PIL import Image
import csv
from typing import List, Any

# --- KONFIGURACJA ---
SIZES: List[int] = [500, 1000, 2000]
STEPS: int = 1000
RATE: float = 0.3
BRIGHTNESS: float = 5.0
DISPLAY_INTERVAL: int = 50
OUTPUT_BASE_DIR: str = "frames_mpi"
RESULTS_FILE: str = "results_mpi_multi.csv"

# --- INICJALIZACJA MPI ---
# Zmienne globalne MPI są akceptowalne w skryptach tego typu
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE_PROC = COMM.Get_size()


def update_local_part(grid: np.ndarray, local_buffer: np.ndarray, rank: int, size_proc: int) -> None:
    """
    Oblicza krok dyfuzji tylko dla wierszy przypisanych do danego procesu (rank).
    Wykorzystuje wektoryzację NumPy.

    Args:
        grid: Pełna siatka z poprzedniego kroku (zsynchronizowana).
        local_buffer: Bufor lokalny, do którego proces wpisuje swoje wyniki.
        rank: Numer procesu.
        size_proc: Całkowita liczba procesów.
    """
    rows = grid.shape[0]
    rows_per_proc = rows // size_proc

    # Wyznaczenie zakresu wierszy dla danego procesu
    start_idx = rank * rows_per_proc
    # Ostatni proces bierze wszystko do końca (obsługa niepodzielnych rozmiarów)
    end_idx = (rank + 1) * rows_per_proc if rank != size_proc - 1 else rows

    # Marginesy bezpieczeństwa (nie obliczamy krawędzi zewnętrznych siatki)
    s = max(1, start_idx)
    e = min(rows - 1, end_idx)

    # Jeśli proces nie ma nic do roboty (np. za dużo procesów na małą siatkę)
    if s >= e:
        local_buffer.fill(0)
        return

    # Definicja wycinków (slices) dla obliczeń Laplasjana
    up = grid[s - 1: e - 1, 1:-1]
    down = grid[s + 1: e + 1, 1:-1]
    left = grid[s: e, 0:-2]
    right = grid[s: e, 2:]
    center = grid[s: e, 1:-1]

    # Obliczenia:
    # 1. Czyścimy bufor lokalny (proces wpisuje wyniki TYLKO w swoim pasie)
    local_buffer.fill(0)

    # 2. Aplikacja wzoru dyfuzji
    local_buffer[s:e, 1:-1] = center + RATE * (up + down + left + right - 4 * center)

    # 3. Clipowanie wartości (ograniczenie 0-1)
    np.clip(local_buffer[s:e, 1:-1], 0, 1, out=local_buffer[s:e, 1:-1])


def save_frame(grid: np.ndarray, step: int, output_dir: str) -> None:
    """Zapisuje aktualny stan siatki do pliku PNG (Wykonuje tylko Rank 0)."""
    frame = np.clip(grid * BRIGHTNESS, 0, 1)
    img = Image.fromarray((frame * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{step:04d}.png"))


def run_simulation_for_size(size: int) -> None:
    """
    Przeprowadza pełną symulację dla jednego rozmiaru siatki.
    Zarządza pętlą czasową, synchronizacją MPI (Allreduce) i zapisem wyników.
    """
    output_dir = os.path.join(OUTPUT_BASE_DIR, str(size))

    # Tylko proces główny tworzy katalogi i wypisuje logi
    if RANK == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- Start MPI size: {size} (Procesy: {SIZE_PROC}) ---")

    # Inicjalizacja siatki
    grid = np.zeros((size, size, 3), dtype=np.float32)
    local_new_grid = np.zeros_like(grid)

    # Warunki początkowe
    grid[size // 2, size // 2] = [1, 0, 0]
    grid[size // 4, size // 4] = [0, 1, 0]
    grid[3 * size // 4, 3 * size // 4] = [0, 0, 1]

    # Synchronizacja przed startem pomiaru
    COMM.Barrier()
    start_time = time.time()

    for step in range(STEPS):
        # 1. Każdy proces liczy swój kawałek
        update_local_part(grid, local_new_grid, RANK, SIZE_PROC)

        # 2. Scalanie wyników: SUMA z buforów lokalnych tworzy pełną nową siatkę
        # (Dlatego local_new_grid musi być zerowane poza obszarem działania procesu)
        COMM.Allreduce(local_new_grid, grid, op=MPI.SUM)

        # 3. Zapis klatki (tylko Rank 0)
        if RANK == 0 and step % DISPLAY_INTERVAL == 0:
            save_frame(grid, step, output_dir)

    # Synchronizacja końcowa i pomiar czasu
    COMM.Barrier()
    elapsed = time.time() - start_time

    if RANK == 0:
        print(f"Zakończono size {size}: {elapsed:.3f}s")
        write_result(size, elapsed)


def write_result(size: int, elapsed: float) -> None:
    """Dopisuje wynik do pliku CSV (tylko Rank 0)."""
    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["size", "processes", "time_s"])
        writer.writerow([size, SIZE_PROC, elapsed])


def main() -> None:
    # Główna pętla po rozmiarach siatki
    for size in SIZES:
        run_simulation_for_size(size)
        COMM.Barrier()


if __name__ == "__main__":
    main()