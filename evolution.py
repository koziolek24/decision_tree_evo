from copy import deepcopy
from evo_tree import TreeNode, cross_breed
import pandas as pd
import random
import numpy as np
import math


def initialize_population(
    population_count: int, parameters_count: int, classes_count: int
) -> list[TreeNode]:
    population: list[TreeNode] = []
    for _ in range(population_count):
        parameterID = random.randint(0, parameters_count - 1)
        classID = random.randint(0, classes_count - 1)
        population.append(
            TreeNode(
                parameter_key=parameterID,
                represented_class=classID,
                parameter_split=random.random(),
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
    train_Y: pd.Series,
    sample_size: int,
    cross_breed_prob: float,
    add_child_prob: float,
    patience: int,
    penalty_rate: float = 0.001,
    shallow_penalty_rate: float = 0.5,
    shallow_threshold: int = 10,
    elitism_rate: float = 0.1,
    penalty_type: str = 'linear'
) -> TreeNode:
    parameters_count = train_X.shape[1]
    classes_count = train_Y.nunique()
    population = initialize_population(
        population_count, parameters_count, classes_count
    )
    best_tree = None
    best_score = -1
    no_improvement_count = 0

    min_val = train_X.min().min()
    max_val = train_X.max().max()

    # Convert to numpy for faster processing
    train_X = train_X.values
    train_Y = train_Y.values

    def rng_class():
        return random.randint(0, classes_count - 1)

    def rng_param():
        return random.randint(0, parameters_count - 1)

    def rng_split():
        return random.uniform(min_val, max_val)

    def rng_delta():
        return random.gauss(0, (max_val - min_val) * 0.25)

    def penalty(tree: TreeNode):
        # Penalize huge trees
        node_count = len(tree.all_nodes())
        
        if penalty_type == 'exponential':
            size_penalty = math.exp(-penalty_rate * node_count)
        else:
            size_penalty = 1 - penalty_rate * node_count
        
        # Penalize very shallow trees heavily (encourage growth)
        if node_count < shallow_threshold:
            return size_penalty * shallow_penalty_rate
            
        return size_penalty

    while True:
        scores: list[float] = []
        for tree in population:
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
            left_winner = max(
                left_sample, key=lambda x: scores[x] * penalty(population[x])
            )
            left_parent = population[left_winner]

            right_sample = random.sample(range(population_count), sample_size)
            right_winner = max(
                right_sample, key=lambda x: scores[x] * penalty(population[x])
            )
            right_parent = population[right_winner]

            if random.random() < cross_breed_prob:
                left_child, right_child = cross_breed(left_parent, right_parent)
            else:
                left_child = deepcopy(left_parent)
                right_child = deepcopy(right_parent)

            left_child.mutate(
                rng_class, rng_param, rng_delta, rng_split, add_child_prob
            )
            right_child.mutate(
                rng_class, rng_param, rng_delta, rng_split, add_child_prob
            )

            new_population.append(left_child)
            if len(new_population) < population_count:
                new_population.append(right_child)
        population = new_population

    assert best_tree is not None
    return best_tree
