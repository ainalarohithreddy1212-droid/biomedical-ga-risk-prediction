import numpy as np

from src.ga_feature_selection import (
    evaluate_features,
    genetic_feature_selection
)


def create_test_data():
    np.random.seed(42)

    X = np.random.rand(40, 6)

    y = np.array(
        [0, 1] * 20
    )

    return X, y


def test_feature_evaluation_returns_valid_score():
    X, y = create_test_data()

    chromosome = np.array(
        [1, 1, 0, 0, 0, 0]
    )

    score = evaluate_features(
        X,
        y,
        chromosome
    )

    assert 0.0 <= score <= 1.0


def test_ga_returns_correct_chromosome_shape():
    X, y = create_test_data()

    chromosome, fitness, history = genetic_feature_selection(
        X,
        y,
        population_size=6,
        generations=3
    )

    assert chromosome.shape == (6,)
    assert 0.0 <= fitness <= 1.0
    assert len(history) == 3


def test_empty_feature_selection_returns_zero():
    X, y = create_test_data()

    chromosome = np.zeros(6)

    score = evaluate_features(
        X,
        y,
        chromosome
    )

    assert score == 0.0
