from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn_wrapper import EvoTreeClassifier
from utils import split_train_test
import numpy as np
import time


def optimize_dataset(dataset_name):
    print(dataset_name)
    train_X, test_X, train_Y, test_Y = split_train_test(dataset_name)
    X = train_X.values
    y = train_Y.values

    if np.isnan(X).any():
        print("Handling NaNs in dataset...")
        X = np.nan_to_num(X, nan=0.0)

    model = EvoTreeClassifier()

    param_dist = {
        "population_count": randint(10, 1000),
        "sample_size": randint(2, 10),
        "cross_breed_prob": uniform(0.6, 0.4),
        "add_child_prob": uniform(0.05, 0.5),
        "patience": randint(10, 100),
        "penalty_rate": loguniform(1e-5, 1e0),
        "elitism_rate": uniform(0.01, 0.2),
        "penalty_type": ["linear", "exponential"],
    }

    n_iter = 25

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    start = time.time()
    search.fit(X, y)
    end = time.time()

    print(f"Optimization finished in {end - start:.2f}s")
    print("Best Parameters:")
    print(search.best_params_)
    print(f"Best CV Score: {search.best_score_:.4f}")

    best_model = search.best_estimator_
    test_score = best_model.score(test_X.values, test_Y.values)
    print(f"Test Set Score: {test_score:.4f}")

    return search.best_params_


if __name__ == "__main__":
    optimize_dataset("breast_cancer")
    optimize_dataset("winequality_red")
    optimize_dataset("winequality_white")
    optimize_dataset("airline_passenger_satisfaction")
