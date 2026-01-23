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
    elitism_rate: float = 0.2,
    penalty_type: str = "linear"
):
    print(f"Loading {dataset_name}")

    try:
        train_X, test_X, train_Y, test_Y = split_train_test(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")

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
        penalty_type=penalty_type
    )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Runtime: {duration:.2f} seconds.")

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
            "add_child_prob": 0.5434434683002587,
            "cross_breed_prob": 0.908897907718663,
            "elitism_rate": 0.04974313630683449,
            "patience": 15,
            "penalty_type": 'linear',
            "population_count": 30,
            "sample_size": 10
        },
        "winequality_red": {
            "add_child_prob": 0.3496,
            "cross_breed_prob": 0.9803,
            "elitism_rate": 0.3687,
            "patience": 20,
            "penalty_type": 'linear',
            "population_count": 50,
            "sample_size": 25
        },
        "winequality_white": {
             "add_child_prob": 0.3560,
             "cross_breed_prob": 0.9933,
             "elitism_rate": 0.2387,
             "patience": 20,
             "penalty_type": 'linear',
             "population_count": 500,
             "sample_size": 5
        },
        "airline_passenger_satisfaction": {
            "add_child_prob": 0.2570,
            "cross_breed_prob": 0.8650,
            "elitism_rate": 0.1627,
            "patience": 50,
            "penalty_type": 'exponential',
            "population_count": 200,
            "sample_size": 100
        },
    }

    for dataset_name, config in configs.items():
        run_experiment(dataset_name, **config)


if __name__ == "__main__":
    main()
