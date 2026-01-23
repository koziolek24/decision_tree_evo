from copy import deepcopy
from evo_tree import TreeNode, cross_breed
import pandas as pd
import random
import numpy as np
import math


def initialize_population(
    population_count: int, parameters_count: int, classes_count: int,
    param_ranges: list[tuple[float, float]]
) -> list[TreeNode]:
    population: list[TreeNode] = []
    for _ in range(population_count):
        parameterID = random.randint(0, parameters_count - 1)
        classID = random.randint(0, classes_count - 1)
        min_val, max_val = param_ranges[parameterID]
        population.append(
            TreeNode(
                parameter_key=parameterID,
                represented_class=classID,
                parameter_split=random.uniform(min_val, max_val),
            )
        )
    return population


def calculate_accuracy(
    tree: TreeNode, X: pd.DataFrame | np.ndarray, Y: pd.Series | np.ndarray
) -> float:
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.Series):
        Y = Y.values

    predictions = tree.predict(X)
    return np.mean(predictions == Y)


def evolution(
    population_count: int,
    train_X: pd.DataFrame,
    train_Y: pd.Series | np.ndarray,
    sample_size: int,
    cross_breed_prob: float,
    add_child_prob: float,
    patience: int,
    penalty_rate: float = 0.001,
    elitism_rate: float = 0.1,
    penalty_type: str = "linear",
) -> TreeNode:
    parameters_count = train_X.shape[1]
    classes_count = len(np.unique(train_Y))
    
    # Ensure sample_size is not larger than population
    sample_size = min(sample_size, population_count)
    
    # Convert to numpy for faster processing
    train_X_values = train_X.values if isinstance(train_X, pd.DataFrame) else train_X
    train_Y_values = train_Y.values if isinstance(train_Y, pd.Series) else train_Y

    # Vectorized param_ranges calculation
    min_vals = np.min(train_X_values, axis=0)
    max_vals = np.max(train_X_values, axis=0)
    param_ranges = list(zip(min_vals, max_vals))

    population = initialize_population(
        population_count, parameters_count, classes_count, param_ranges
    )
    best_tree = None
    best_score = -1
    no_improvement_count = 0

    def rng_class():
        return random.randint(0, classes_count - 1)

    def rng_param():
        return random.randint(0, parameters_count - 1)

    def rng_delta(param_key: int):
        min_val, max_val = param_ranges[param_key]
        return random.gauss(0, (max_val - min_val) * 0.25)

    def penalty(tree: TreeNode):
        node_count = len(tree.all_nodes())

        if penalty_type == "exponential":
            size_penalty = math.exp(-penalty_rate * node_count)
        else:
            size_penalty = max(0.2, 1 - penalty_rate * node_count)

        return size_penalty

    while True:
        scores: list[float] = []
        for tree in population:
            if tree.last_score > -1:
                score = tree.last_score
            else:
                score = calculate_accuracy(tree, train_X, train_Y)
                tree.last_score = score
            scores.append(score)

            if score > best_score:
                best_score = score
                best_tree = tree
                no_improvement_count = 0

        if no_improvement_count >= patience:
            break
        no_improvement_count += 1

        new_population: list[TreeNode] = sorted(
            population, key=lambda x: x.last_score * penalty(x), reverse=True
        )[: int(len(population) * elitism_rate)]

        while len(new_population) < population_count:
            left_sample = random.sample(range(population_count), sample_size)
            left_winner_index = max(
                left_sample, key=lambda x: scores[x] * penalty(population[x])
            )
            left_parent = population[left_winner_index]

            right_sample = random.sample(range(population_count), sample_size)
            right_winner_index = max(
                right_sample, key=lambda x: scores[x] * penalty(population[x])
            )
            right_parent = population[right_winner_index]

            if random.random() < cross_breed_prob:
                left_child, right_child = cross_breed(left_parent, right_parent)
            else:
                left_child = deepcopy(left_parent)
                right_child = deepcopy(right_parent)

            left_child.mutate(rng_class, rng_param, param_ranges, add_child_prob)
            right_child.mutate(rng_class, rng_param, param_ranges, add_child_prob)

            left_child.last_score = -1.0
            right_child.last_score = -1.0

            new_population.append(left_child)
            if len(new_population) < population_count:
                new_population.append(right_child)
        population = new_population

    assert best_tree is not None
    return best_tree
