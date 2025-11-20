import subprocess
import sys
import os
import csv
import time
import importlib.util
from typing import List, Dict, Optional

# --- KONFIGURACJA ---
SCRIPTS: Dict[str, str] = {
    "numpy": "diffusion_numpy.py",
    "numba": "diffusion_numba.py",
    "mpi": "diffusion_mpi.py",
    "cupy": "diffusion_cupy.py",
    "plot": "benchmark_plot.py"
}

MPI_RESULTS_FILE: str = "results_mpi_multi.csv"
MPI_PROCS_TO_TEST: List[int] = [1, 2, 4]

# Mapowanie: nazwa pakietu pip -> nazwa modułu do importu
REQUIRED_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
    "numba": "numba",
    "mpi4py": "mpi4py",
    "matplotlib": "matplotlib",
    "Pillow": "PIL"
}


def install_package(package_name: str) -> None:
    """Instaluje pakiet pip w bieżącym środowisku Pythona."""
    print(f"   [INSTALACJA] Pobieranie {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"   [SUKCES] Zainstalowano {package_name}.")
    except subprocess.CalledProcessError:
        print(f"   [BŁĄD] Nie udało się zainstalować {package_name}. Sprawdź uprawnienia.")
        sys.exit(1)


def check_environment() -> bool:
    """
    Weryfikuje obecność wymaganych bibliotek oraz narzędzia mpiexec.
    Zwraca True, jeśli środowisko MPI jest dostępne, w przeciwnym razie False.
    """
    print("===================================================")
    print("   SPRAWDZANIE ŚRODOWISKA I BIBLIOTEK")
    print("===================================================")

    # 1. Weryfikacja pakietów CPU (i auto-instalacja)
    for package, import_name in REQUIRED_PACKAGES.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"   [BRAK] Biblioteka '{import_name}' nie znaleziona.")
            install_package(package)
        else:
            print(f"   [OK] Biblioteka '{import_name}' jest dostępna.")

    # 2. Weryfikacja CuPy (GPU) - tylko ostrzeżenie
    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        print("   [INFO] Brak biblioteki 'cupy' (GPU). Testy GPU zostaną pominięte.")
    else:
        print("   [OK] Biblioteka 'cupy' jest dostępna.")

    # 3. Weryfikacja binarki MPI (Systemowe)
    mpi_available = False
    try:
        subprocess.run(["mpiexec", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   [OK] Narzędzie systemowe 'mpiexec' jest dostępne.")
        mpi_available = True
    except FileNotFoundError:
        print("   [UWAGA] Nie znaleziono 'mpiexec' w PATH. Testy MPI nie zadziałają.")

    return mpi_available


def run_process(cmd: List[str], description: str) -> bool:
    """Uruchamia proces zewnętrzny i mierzy jego czas całkowity."""
    print(f"\n>>> {description}...")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"    [OK] Zakończono pomyślnie w {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError:
        print(f"    [BŁĄD] Proces zakończył się kodem błędu.")
        return False
    except FileNotFoundError:
        print(f"    [BŁĄD] Nie znaleziono pliku lub polecenia: {cmd[0]}")
        return False


def run_mpi_suite() -> None:
    """Specjalna procedura dla MPI: reset pliku CSV i uruchomienie dla wielu procesów."""
    if not os.path.exists(SCRIPTS["mpi"]):
        print("Pominięto MPI - brak pliku skryptu.")
        return

    print(f"\n>>> Rozpoczynanie serii testów MPI...")
    try:
        # Inicjalizacja pustego pliku CSV z nagłówkiem
        with open(MPI_RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["size", "processes", "time_s"])

        # Uruchamianie dla zadanej liczby procesów
        for n in MPI_PROCS_TO_TEST:
            cmd = ["mpiexec", "-n", str(n), sys.executable, SCRIPTS["mpi"]]
            success = run_process(cmd, f"MPI (Procesy: {n})")
            if not success:
                print("    Przerwano testy MPI z powodu błędu.")
                break
    except Exception as e:
        print(f"    Błąd krytyczny podczas testów MPI: {e}")


def main() -> None:
    # 1. Weryfikacja środowiska
    mpi_ready = check_environment()

    print("\n===================================================")
    print("   START BENCHMARKU DYFUZJI")
    print("===================================================")

    # 2. NumPy (Baseline)
    if os.path.exists(SCRIPTS["numpy"]):
        run_process([sys.executable, SCRIPTS["numpy"]], "NumPy Benchmark")

    # 3. Numba (Multithreading)
    if os.path.exists(SCRIPTS["numba"]):
        run_process([sys.executable, SCRIPTS["numba"]], "Numba Benchmark")

    # 4. MPI (Multiprocessing)
    if mpi_ready:
        run_mpi_suite()
    else:
        print("\n>>> Pominięto testy MPI (brak środowiska).")

    # 5. CuPy (GPU)
    if os.path.exists(SCRIPTS["cupy"]):
        if importlib.util.find_spec("cupy"):
            run_process([sys.executable, SCRIPTS["cupy"]], "CuPy Benchmark (GPU)")
        else:
            print("\n>>> Pominięto CuPy - brak zainstalowanej biblioteki.")

    # 6. Generowanie wykresów
    if os.path.exists(SCRIPTS["plot"]):
        run_process([sys.executable, SCRIPTS["plot"]], "Generowanie wykresów i raportu")
    else:
        print("\n[UWAGA] Brak skryptu do wykresów.")

    print("\n===================================================")
    print("   KONIEC")
    print("===================================================")


if __name__ == "__main__":
    main()