"""
Genetic Algorithm based biomedical feature selection.

This module searches for useful subsets of biomedical features
using selection, crossover, and mutation.
"""

from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    chromosome: np.ndarray
) -> float:
    """
    Evaluate a feature subset using Logistic Regression.

    Parameters
    ----------
    X : np.ndarray
        Biomedical feature matrix.
    y : np.ndarray
        Binary risk labels.
    chromosome : np.ndarray
        Binary vector indicating selected features.

    Returns
    -------
    float
        Validation accuracy of the selected feature subset.
    """

    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return 0.0

    X_selected = X[:, selected]

    X_train, X_val, y_train, y_val = train_test_split(
        X_selected,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return float(accuracy_score(y_val, predictions))


def genetic_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 10,
    generations: int = 10,
    mutation_rate: float = 0.1
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Perform Genetic Algorithm feature selection.

    Parameters
    ----------
    X : np.ndarray
        Biomedical feature matrix.
    y : np.ndarray
        Target risk labels.
    population_size : int
        Number of chromosomes in the population.
    generations : int
        Number of evolutionary generations.
    mutation_rate : float
        Probability of mutation.

    Returns
    -------
    Tuple[np.ndarray, float, List[float]]
        Best chromosome, best fitness and fitness history.
    """

    number_of_features = X.shape[1]

    population = np.random.randint(
        0,
        2,
        size=(population_size, number_of_features)
    )

    best_chromosome = population[0]
    best_fitness = 0.0
    fitness_history: List[float] = []

    for _ in range(generations):

        fitness_values = np.array([
            evaluate_features(X, y, chromosome)
            for chromosome in population
        ])

        best_index = int(np.argmax(fitness_values))

        if fitness_values[best_index] > best_fitness:
            best_fitness = float(fitness_values[best_index])
            best_chromosome = population[best_index].copy()

        fitness_history.append(best_fitness)

        # Select top half
        selected_indices = np.argsort(fitness_values)[
            -population_size // 2:
        ]

        parents = population[selected_indices]

        new_population = list(parents)

        while len(new_population) < population_size:

            parent1 = parents[
                np.random.randint(len(parents))
            ]

            parent2 = parents[
                np.random.randint(len(parents))
            ]

            crossover_point = np.random.randint(
                1,
                number_of_features
            )

            child = np.concatenate([
                parent1[:crossover_point],
                parent2[crossover_point:]
            ])

            mutation_mask = (
                np.random.random(number_of_features)
                < mutation_rate
            )

            child[mutation_mask] = (
                1 - child[mutation_mask]
            )

            new_population.append(child)

        population = np.array(new_population)

    return best_chromosome, best_fitness, fitness_history
