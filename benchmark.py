"""
Benchmark the Genetic Algorithm feature-selection runtime.
"""

import time

import pandas as pd

from src.ga_feature_selection import (
    genetic_feature_selection
)


def main() -> None:
    """Measure GA execution time on the biomedical dataset."""

    data = pd.read_csv(
        "data/patient_data.csv"
    )

    X = data.drop(
        "risk",
        axis=1
    ).values

    y = data["risk"].values

    start_time = time.perf_counter()

    chromosome, fitness, history = (
        genetic_feature_selection(
            X,
            y,
            population_size=10,
            generations=10
        )
    )

    elapsed = time.perf_counter() - start_time

    print("=" * 50)
    print("GA PERFORMANCE BENCHMARK")
    print("=" * 50)

    print(
        f"Patients processed : {len(data)}"
    )

    print(
        f"Features processed : {X.shape[1]}"
    )

    print(
        f"Execution time     : {elapsed * 1000:.2f} ms"
    )

    print(
        f"Best fitness       : {fitness:.4f}"
    )

    print(
        f"Generations        : {len(history)}"
    )

    print(
        "Selected features  :",
        chromosome
    )


if __name__ == "__main__":
    main()
