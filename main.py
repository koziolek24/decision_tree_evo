from concurrent.futures import ProcessPoolExecutor
import statistics
import numpy as np
import pandas as pd
from sklearn.externals.array_api_compat.numpy import ndarray
import sklearn.metrics
from evo_tree import Tree
from utils import split_train_test
from evolution import evolution, calculate_accuracy
import time


def calculate_f1_score(
    tree: Tree, X: pd.DataFrame | np.ndarray, Y: pd.Series | np.ndarray
) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.values  # type: ignore
    if isinstance(Y, pd.Series):
        Y = Y.values  # type: ignore

    predictions = tree.predict(X)
    return np.asarray(sklearn.metrics.f1_score(Y, predictions, average="micro"))


def calculate_confusion_matrix(
    tree: Tree, X: pd.DataFrame | np.ndarray, Y: pd.Series | np.ndarray
) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.values  # type: ignore
    if isinstance(Y, pd.Series):
        Y = Y.values  # type: ignore

    predictions = tree.predict(X)
    return np.asarray(sklearn.metrics.confusion_matrix(Y, predictions))


def run_one(
    dataset_name: str,
    population_count: int = 200,
    sample_size: int = 5,
    cross_breed_prob: float = 0.8,
    add_child_prob: float = 0.6,
    patience: int = 10,
    elitism_rate: float = 0.2,
    penalty_type: str = "linear",
    penalty_rate: float = 0.1,
):
    try:
        train_X, test_X, train_Y, test_Y = split_train_test(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    start_time = time.time()

    best_tree = evolution(
        population_count=population_count,
        train_X=train_X,
        train_Y=train_Y,
        sample_size=sample_size,
        cross_breed_prob=cross_breed_prob,
        add_child_prob=add_child_prob,
        patience=patience,
        elitism_rate=elitism_rate,
        penalty_type=penalty_type,
        penalty_rate=penalty_rate,
    )
    end_time = time.time()
    duration = end_time - start_time
    return (
        duration,
        calculate_accuracy(best_tree, train_X, train_Y),
        calculate_accuracy(best_tree, test_X, test_Y),
        calculate_f1_score(best_tree, test_X, test_Y),
        calculate_confusion_matrix(best_tree, test_X, test_Y),
    )


def run_experiment(
    dataset_name: str,
    population_count: int = 200,
    sample_size: int = 5,
    cross_breed_prob: float = 0.8,
    add_child_prob: float = 0.6,
    patience: int = 10,
    elitism_rate: float = 0.2,
    penalty_type: str = "linear",
    penalty_rate: float = 0.1,
):
    print(f"Loading {dataset_name}")

    try:
        train_X, test_X, _, _ = split_train_test(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")

    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(
                run_one,
                dataset_name=dataset_name,
                population_count=population_count,
                sample_size=sample_size,
                cross_breed_prob=cross_breed_prob,
                add_child_prob=add_child_prob,
                patience=patience,
                elitism_rate=elitism_rate,
                penalty_type=penalty_type,
                penalty_rate=penalty_rate,
            )
            for _ in range(128)
        ]
        trees = [res for future in futures if (res := future.result()) is not None]

    # print([tree[4].shape for tree in trees])

    print(f"Runtime: {sum(tree[0] for tree in trees):.2f} seconds.")
    print(f"Train Accuracy: {statistics.mean(tree[1] for tree in trees):.4f}")
    print(f"Test Accuracy:  {statistics.mean(tree[2] for tree in trees):.4f}")
    print(f"F1 scores: {np.mean([tree[3] for tree in trees], axis=0)}")
    print(f"Confution Matrix:\n{np.sum([tree[4] for tree in trees], axis=0)}")

    #     return {
    #         "dataset": dataset_name,
    #         "train_accuracy": train_accuracy,
    #         "test_accuracy": test_accuracy,
    #         "duration": duration,
    #         "population_count": population_count,
    #         "sample_size": sample_size,
    #         "cross_breed_prob": cross_breed_prob,
    #         "add_child_prob": add_child_prob,
    #     }
    # else:
    #     print("Evolution failed to produce a valid tree.")
    #     return None


def main():
    configs = {
        "breast_cancer": {
            "add_child_prob": np.float64(0.36249280112905485),
            "cross_breed_prob": np.float64(0.7214175617073775),
            "elitism_rate": np.float64(0.19526835905210316),
            "patience": 38,
            "penalty_rate": np.float64(0.0012836385629361605),
            "penalty_type": "linear",
            "population_count": 731,
            "sample_size": 6,
        },
        "winequality_red": {
            "add_child_prob": np.float64(0.49337946181018266),
            "cross_breed_prob": np.float64(0.9791928857189796),
            "elitism_rate": np.float64(0.14775043643394817),
            "patience": 11,
            "penalty_rate": np.float64(0.0005024655310849527),
            "penalty_type": "exponential",
            "population_count": 20,
            "sample_size": 2,
        },
        "winequality_white": {
            "add_child_prob": np.float64(0.36777916567229524),
            "cross_breed_prob": np.float64(0.6511647015928227),
            "elitism_rate": np.float64(0.08181593866697326),
            "patience": 99,
            "penalty_rate": np.float64(0.00023740776320511935),
            "penalty_type": "exponential",
            "population_count": 562,
            "sample_size": 3,
        },
        "airline_passenger_satisfaction": {
            "add_child_prob": np.float64(0.139914760771449),
            "cross_breed_prob": np.float64(0.7355521458621909),
            "elitism_rate": np.float64(0.14826178247850977),
            "patience": 46,
            "penalty_rate": np.float64(7.790078746503111e-05),
            "penalty_type": "exponential",
            "population_count": 370,
            "sample_size": 3,
        },
    }

    for dataset_name, config in configs.items():
        run_experiment(dataset_name, **config)


if __name__ == "__main__":
    main()
