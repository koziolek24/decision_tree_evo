from copy import deepcopy
from evo_tree import TreeNode, cross_breed
import pandas as pd
import random


def initialize_population(
    population_count: int, parameters_count: int, classes_count: int
) -> list[TreeNode]:
    population: list[TreeNode] = []
    for _ in range(population_count):
        parameterID = random.randint(0, parameters_count - 1)
        classID = random.randint(0, classes_count - 1)
        population.append(TreeNode(parameterID, [], classID))
    return population


def calculate_accuracy(tree: TreeNode, X: pd.DataFrame, Y: pd.Series) -> float:
    correct = 0
    for row, target in zip(X.itertuples(index=False, name=None), Y):
        prediction = tree.get_class(row)  # type: ignore
        if prediction == target:
            correct += 1

    return correct / len(Y)


def evolution(
    generations: int,
    population_count: int,
    train_X: pd.DataFrame,
    train_Y: pd.Series,
    sample_size: int,
    cross_breed_prob: float,
    add_child_prob: float,
) -> TreeNode:
    parameters_count = train_X.shape[1]
    classes_count = train_Y.nunique()
    population = initialize_population(
        population_count, parameters_count, classes_count
    )
    best_tree = None
    best_score = -1

    min_val = train_X.min().min()
    max_val = train_X.max().max()

    def rng_class():
        return random.randint(0, classes_count - 1)

    def rng_param():
        return random.randint(0, parameters_count - 1)

    def rng_split():
        return random.uniform(min_val, max_val)

    def rng_delta():
        return random.gauss(0, (max_val - min_val) * 0.1)

    def penalty(tree: TreeNode):
        return 1 - 0.01 * len(tree.all_nodes())

    for _ in range(generations):
        scores: list[float] = []
        for tree in population:
            score = calculate_accuracy(tree, train_X, train_Y)
            tree.last_score = score
            scores.append(score)

            if score > best_score:
                best_score = score
                best_tree = tree

        new_population: list[TreeNode] = sorted(
            population, key=lambda x: x.last_score * penalty(x)
        )[: int(len(population) * 0.10)]
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
