import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
import csv
import collections
import os
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set

# Definicje typów dla lepszej czytelności
# Słownik zagnieżdżony: liczba wątków/procesów -> rozmiar -> czas
NestedResults = Dict[int, Dict[int, float]]
# Słownik płaski: rozmiar -> czas
FlatResults = Dict[int, float]
# Główna struktura danych
BenchmarkData = Dict[str, Union[NestedResults, FlatResults]]

FILES = {
    "numpy": "results_numpy.csv",
    "numba": "results_numba.csv",
    "mpi": "results_mpi_multi.csv",
    "cupy": "results_cupy.csv"
}

OUTPUT_DIR = "wykresy_sprawozdanie"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data() -> BenchmarkData:
    """
    Wczytuje dane z plików CSV do słownika z podziałem na metody.
    Struktura 'numba' i 'mpi' jest zagnieżdżona (wątki/procesy),
    natomiast 'numpy' i 'cupy' są płaskie (tylko rozmiar).
    """
    data: BenchmarkData = {
        "numpy": {},
        "numba": collections.defaultdict(dict),
        "mpi": collections.defaultdict(dict),
        "cupy": {}
    }

    if os.path.exists(FILES["numpy"]):
        with open(FILES["numpy"]) as f:
            for row in csv.DictReader(f):
                data["numpy"][int(row["size"])] = float(row["time_s"])

    if os.path.exists(FILES["numba"]):
        with open(FILES["numba"]) as f:
            for row in csv.DictReader(f):
                # numba: threads -> size -> time
                data["numba"][int(row["threads"])][int(row["size"])] = float(row["time_s"])

    if os.path.exists(FILES["mpi"]):
        with open(FILES["mpi"]) as f:
            for row in csv.DictReader(f):
                # mpi: processes -> size -> time
                data["mpi"][int(row["processes"])][int(row["size"])] = float(row["time_s"])

    if os.path.exists(FILES["cupy"]):
        with open(FILES["cupy"]) as f:
            for row in csv.DictReader(f):
                data["cupy"][int(row["size"])] = float(row["time_s"])

    return data


def get_common_sizes(data: BenchmarkData) -> List[int]:
    """Zwraca posortowaną listę rozmiarów siatki występujących w wynikach."""
    sizes: Set[int] = set(data["numpy"].keys())
    if data["cupy"]:
        sizes.update(data["cupy"].keys())
    if data["numba"]:
        # Pobieramy klucze z pierwszego dostępnego zestawu wątków
        first_thread_key = list(data["numba"].keys())[0]
        sizes.update(data["numba"][first_thread_key].keys())
    return sorted(list(sizes))


def add_line_labels(x: List[float], y: List[float], ax: Optional[Axes] = None) -> None:
    """Dodaje etykiety wartości nad punktami wykresu liniowego."""
    if ax is None:
        ax = plt.gca()
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=9, fontweight='bold')


def add_bar_labels(rects: BarContainer, ax: Optional[Axes] = None) -> None:
    """Dodaje etykiety wartości nad słupkami wykresu."""
    if ax is None:
        ax = plt.gca()
    ax.bar_label(rects, fmt='%.2f', padding=3, fontsize=9)


def plot_scaling(data_nested: NestedResults, sizes: List[int], method_name: str, x_label: str) -> None:
    """
    Generuje dwa wykresy dla metod równoległych (MPI/Numba):
    1. Speedup (Przyśpieszenie)
    2. Efficiency (Efektywność)
    """
    if not data_nested:
        return

    threads_list = sorted(data_nested.keys())

    # --- Wykres 1: Speedup ---
    plt.figure(figsize=(11, 7))
    max_speedup = 0.0

    for size in sizes:
        # Speedup liczymy względem wykonania na 1 wątku/procesie tej samej metody
        if 1 not in data_nested or size not in data_nested[1]:
            continue

        t1 = data_nested[1][size]
        x_vals, y_vals = [], []

        for th in threads_list:
            if size in data_nested[th]:
                val = t1 / data_nested[th][size]
                x_vals.append(th)
                y_vals.append(val)
                if val > max_speedup:
                    max_speedup = val

        plt.plot(x_vals, y_vals, marker='o', linewidth=2, label=f"Size {size}")
        add_line_labels(x_vals, y_vals)

    # Linia idealnego przyśpieszenia liniowego
    plt.plot(threads_list, threads_list, 'k--', alpha=0.3, label="Ideal")

    plt.title(f"{method_name}: Przyśpieszenie (Speedup)", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Przyśpieszenie (krotność)", fontsize=12)
    plt.xticks(threads_list)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.ylim(0, max(max_speedup * 1.1, max(threads_list) + 0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scaling_speedup_{method_name.lower()}.png")
    plt.close()

    # --- Wykres 2: Efficiency ---
    plt.figure(figsize=(11, 7))
    max_efficiency = 0.0

    for size in sizes:
        if 1 not in data_nested or size not in data_nested[1]:
            continue

        t1 = data_nested[1][size]
        x_vals, y_vals = [], []

        for th in threads_list:
            if size in data_nested[th]:
                speedup = t1 / data_nested[th][size]
                efficiency = (speedup / th) * 100
                x_vals.append(th)
                y_vals.append(efficiency)
                if efficiency > max_efficiency:
                    max_efficiency = efficiency

        plt.plot(x_vals, y_vals, marker='s', linewidth=2, label=f"Size {size}")
        add_line_labels(x_vals, y_vals)

    plt.axhline(100, color='k', linestyle='--', alpha=0.3)
    plt.title(f"{method_name}: Efektywność zrównoleglenia", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Efektywność [%]", fontsize=12)
    plt.xticks(threads_list)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.ylim(0, max(110.0, max_efficiency + 15))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scaling_efficiency_{method_name.lower()}.png")
    plt.close()


def plot_best_comparison(data: BenchmarkData, sizes: List[int]) -> None:
    """
    Porównuje surowe czasy wykonania dla wszystkich metod.
    Dla metod równoległych (MPI, Numba) wybiera najlepszy czas (najwięcej wątków).
    """
    plt.figure(figsize=(12, 8))
    bar_width = 0.2
    x = np.arange(len(sizes))

    # Pobieranie czasów (default 0 jeśli brak danych)
    times_numpy = [data["numpy"].get(s, 0) for s in sizes]

    times_numba = []
    times_mpi = []

    for s in sizes:
        # Wyciągamy minimalny czas ze wszystkich konfiguracji wątków
        numba_data: NestedResults = data["numba"]
        n_vals = [numba_data[th][s] for th in numba_data if s in numba_data[th]]
        times_numba.append(min(n_vals) if n_vals else 0)

        mpi_data: NestedResults = data["mpi"]
        m_vals = [mpi_data[p][s] for p in mpi_data if s in mpi_data[p]]
        times_mpi.append(min(m_vals) if m_vals else 0)

    times_cupy = [data["cupy"].get(s, 0) for s in sizes]

    # Rysowanie słupków
    rects1 = plt.bar(x - 1.5 * bar_width, times_numpy, bar_width, label='NumPy', color='gray')
    rects2 = plt.bar(x - 0.5 * bar_width, times_numba, bar_width, label='Numba (Best)', color='royalblue')
    rects3 = plt.bar(x + 0.5 * bar_width, times_mpi, bar_width, label='MPI (Best)', color='forestgreen')
    rects4 = plt.bar(x + 1.5 * bar_width, times_cupy, bar_width, label='CuPy (GPU)', color='crimson')

    plt.xlabel('Rozmiar siatki', fontsize=12)
    plt.ylabel('Czas [s]', fontsize=12)
    plt.title('Porównanie czasów wykonania (Mniej = Lepiej)', fontsize=14)
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    for r in [rects1, rects2, rects3, rects4]:
        add_bar_labels(r)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_best_times.png")
    plt.close()


def plot_relative_speedup_all(data: BenchmarkData, sizes: List[int]) -> None:
    """
    Pokazuje przyśpieszenie wszystkich metod względem referencyjnego NumPy.
    Oś Y jest logarytmiczna, aby obsłużyć duże różnice (np. GPU).
    """
    plt.figure(figsize=(12, 8))

    speedup_numba, speedup_mpi, speedup_cupy = [], [], []
    valid_sizes = []

    for s in sizes:
        if s not in data["numpy"]:
            continue

        t_base = data["numpy"][s]
        valid_sizes.append(s)

        # Numba speedup
        numba_data: NestedResults = data["numba"]
        n_times = [numba_data[th][s] for th in numba_data if s in numba_data[th]]
        t_numba = min(n_times) if n_times else t_base
        speedup_numba.append(t_base / t_numba)

        # MPI speedup
        mpi_data: NestedResults = data["mpi"]
        m_times = [mpi_data[p][s] for p in mpi_data if s in mpi_data[p]]
        t_mpi = min(m_times) if m_times else t_base
        speedup_mpi.append(t_base / t_mpi)

        # CuPy speedup
        if s in data["cupy"]:
            speedup_cupy.append(t_base / data["cupy"][s])
        else:
            speedup_cupy.append(0)

    x = np.arange(len(valid_sizes))
    width = 0.25

    r0 = plt.bar(x - width, [1] * len(valid_sizes), width, label='NumPy (Baza)', color='lightgray')
    r1 = plt.bar(x, speedup_numba, width, label='Numba', color='royalblue')
    r2 = plt.bar(x + width, speedup_mpi, width, label='MPI', color='forestgreen')

    has_cupy = any(val > 0 for val in speedup_cupy)
    if has_cupy:
        r3 = plt.bar(x + 2 * width, speedup_cupy, width, label='CuPy (GPU)', color='crimson')

    plt.xlabel('Rozmiar siatki', fontsize=12)
    plt.ylabel('Przyśpieszenie względem NumPy (krotność)', fontsize=12)
    plt.title('Ile razy szybciej niż zwykły NumPy?', fontsize=14)
    plt.xticks(x, valid_sizes)
    plt.legend(loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)

    # Skala logarytmiczna z ludzkim formatowaniem (10, 100 zamiast 10^1, 10^2)
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    add_bar_labels(r0)
    add_bar_labels(r1)
    add_bar_labels(r2)
    if has_cupy:
        add_bar_labels(r3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/relative_speedup_all.png")
    plt.close()


def main() -> None:
    data = load_data()
    sizes = get_common_sizes(data)

    if not sizes:
        print("[ERROR] Brak danych do przetworzenia. Upewnij się, że pliki .csv istnieją.")
        return

    # Generowanie wykresów
    # rzutowanie typów (cast) jest tu domyślne dzięki definicji NestedResults,
    # ale dla mypy data["numba"] może być Union, więc w ścisłym trybie wymagałoby checku.
    # Tutaj zakładamy poprawność struktury z load_data.

    plot_scaling(data["numba"], sizes, "Numba", "Liczba wątków")
    plot_scaling(data["mpi"], sizes, "MPI", "Liczba procesów")
    plot_best_comparison(data, sizes)
    plot_relative_speedup_all(data, sizes)


if __name__ == "__main__":
    main()