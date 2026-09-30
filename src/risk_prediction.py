import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/patient_data.csv")

print("Dataset loaded successfully!")
print("Number of patients:", len(data))

print("\nAvailable features:")
print(list(data.columns[:-1]))


# ==========================================
# 2. PREPARE DATA
# ==========================================

X = data.drop("risk", axis=1)
y = data["risk"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 3. FITNESS FUNCTION
# ==========================================

def calculate_fitness(individual):

    selected_features = np.array(individual, dtype=bool)

    # Avoid individuals that select no features
    if not selected_features.any():
        return 0

    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, predictions)

    # Small penalty for selecting too many features
    feature_penalty = sum(individual) / len(individual)

    fitness = accuracy - (0.01 * feature_penalty)

    return fitness


# ==========================================
# 4. CREATE INITIAL POPULATION
# ==========================================

def create_population(population_size):

    population = []

    for _ in range(population_size):

        individual = [
            random.randint(0, 1)
            for _ in range(len(feature_names))
        ]

        # Make sure at least one feature is selected
        if sum(individual) == 0:
            individual[random.randint(0, len(individual) - 1)] = 1

        population.append(individual)

    return population


# ==========================================
# 5. SELECTION
# ==========================================

def selection(population, fitness_scores):

    sorted_population = [
        individual
        for _, individual in sorted(
            zip(fitness_scores, population),
            reverse=True
        )
    ]

    # Keep the best half
    return sorted_population[:len(population) // 2]


# ==========================================
# 6. CROSSOVER
# ==========================================

def crossover(parent1, parent2):

    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


# ==========================================
# 7. MUTATION
# ==========================================

def mutation(individual, mutation_rate=0.1):

    mutated = individual.copy()

    for i in range(len(mutated)):

        if random.random() < mutation_rate:

            mutated[i] = 1 - mutated[i]

    # Ensure at least one feature is selected
    if sum(mutated) == 0:
        mutated[random.randint(0, len(mutated) - 1)] = 1

    return mutated


# ==========================================
# 8. GENETIC ALGORITHM
# ==========================================

def genetic_algorithm(
    population_size=20,
    generations=10
):

    population = create_population(population_size)

    best_individual = None
    best_fitness = 0

    print("\n==========================================")
    print("GENETIC ALGORITHM STARTED")
    print("==========================================")

    for generation in range(generations):

        fitness_scores = [
            calculate_fitness(individual)
            for individual in population
        ]

        current_best_index = np.argmax(fitness_scores)

        current_best = population[current_best_index]
        current_fitness = fitness_scores[current_best_index]

        if current_fitness > best_fitness:

            best_fitness = current_fitness
            best_individual = current_best.copy()

        print(
            f"Generation {generation + 1}: "
            f"Best Fitness = {current_fitness:.4f}"
        )

        # Selection
        selected = selection(
            population,
            fitness_scores
        )

        # Create next generation
        next_generation = selected.copy()

        while len(next_generation) < population_size:

            parent1, parent2 = random.sample(selected, 2)

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutation(child1)
            child2 = mutation(child2)

            next_generation.append(child1)

            if len(next_generation) < population_size:
                next_generation.append(child2)

        population = next_generation

    return best_individual, best_fitness


# ==========================================
# 9. RUN GENETIC ALGORITHM
# ==========================================

best_features, best_fitness = genetic_algorithm()


# ==========================================
# 10. DISPLAY BEST FEATURES
# ==========================================

selected_features = [
    feature_names[i]
    for i in range(len(feature_names))
    if best_features[i] == 1
]

print("\n==========================================")
print("OPTIMAL FEATURE SELECTION")
print("==========================================")

print("Selected features:")

for feature in selected_features:
    print("✓", feature)

print("\nBest fitness:", round(best_fitness, 4))


# ==========================================
# 11. FINAL RISK MODEL
# ==========================================

feature_indices = [
    i
    for i in range(len(feature_names))
    if best_features[i] == 1
]

X_train_final = X_train_scaled[:, feature_indices]
X_test_final = X_test_scaled[:, feature_indices]

final_model = LogisticRegression(max_iter=1000)

final_model.fit(
    X_train_final,
    y_train
)

final_predictions = final_model.predict(
    X_test_final
)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)


print("\n==========================================")
print("FINAL MODEL RESULTS")
print("==========================================")

print(
    "Final Model Accuracy:",
    round(final_accuracy * 100, 2),
    "%"
)


# ==========================================
# 12. NEW PATIENT PREDICTION
# ==========================================

new_patient = pd.DataFrame(
    [[60, 155, 100, 92, 38.0, 175]],
    columns=feature_names
)
new_patient_scaled = scaler.transform(
    new_patient
)

new_patient_selected = (
    new_patient_scaled[:, feature_indices]
)

risk_prediction = final_model.predict(
    new_patient_selected
)[0]

risk_probability = final_model.predict_proba(
    new_patient_selected
)[0][1]


print("\n==========================================")
print("NEW PATIENT RISK ASSESSMENT")
print("==========================================")

if risk_prediction == 1:

    print("Risk Level: HIGH")

else:

    print("Risk Level: LOW")

print(
    "Estimated Risk Probability:",
    round(risk_probability * 100, 2),
    "%"
)

print("==========================================")
