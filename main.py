from utils import split_train_test
from evolution import evolution, calculate_accuracy
import time


def run_experiment(
    dataset_name: str,
    population_count: int = 200,
    sample_size: int = 5,
    cross_breed_prob: float = 0.8,
    add_child_prob: float = 0.6,
    patience: int = 10,
):
    print(f"\n{'=' * 20} {dataset_name} {'=' * 20}")
    print(f"Loading {dataset_name} dataset...")

    try:
        train_X, test_X, train_Y, test_Y = split_train_test(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")

    print("Starting evolution...")
    start_time = time.time()

    best_tree = evolution(
        population_count=population_count,
        train_X=train_X,
        train_Y=train_Y,
        sample_size=sample_size,
        cross_breed_prob=cross_breed_prob,
        add_child_prob=add_child_prob,
        patience=patience,
    )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Evolution finished in {duration:.2f} seconds.")

    if best_tree:
        test_accuracy = calculate_accuracy(best_tree, test_X, test_Y)
        train_accuracy = calculate_accuracy(best_tree, train_X, train_Y)

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:  {test_accuracy:.4f}")

        return {
            "dataset": dataset_name,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "duration": duration,
            "population_count": population_count,
            "sample_size": sample_size,
            "cross_breed_prob": cross_breed_prob,
            "add_child_prob": add_child_prob,
        }
    else:
        print("Evolution failed to produce a valid tree.")
        return None


def main():
    configs = {
        "breast_cancer": {
            "population_count": 200,
            "patience": 10,
            "sample_size": 5
        },
        "winequality_red": {
            "population_count": 300,
            "patience": 15,
            "sample_size": 10
        },
        "winequality_white": {
            "population_count": 300,
            "patience": 15,
            "sample_size": 10
        },
        "airline_passenger_satisfaction": {
            "population_count": 500,
            "patience": 20,
            "sample_size": 5,
            "elitism_rate": 0.2,
            "cross_breed_prob": 1.0,
            "add_child_prob": 0.1,
            "penalty_type": "linear",
            "shallow_penalty_rate": 0.1,
            "shallow_threshold": 15
        },
    }

    for dataset_name, config in configs.items():
        run_experiment(dataset_name, **config)


if __name__ == "__main__":
    main()
