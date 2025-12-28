from utils import split_train_test
from evolution import evolution, calculate_accuracy
import time

def run_experiment(dataset_name):
    print(f"\n{'='*20} {dataset_name} {'='*20}")
    print(f"Loading {dataset_name} dataset...")
    
    try:
        train_X, test_X, train_Y, test_Y = split_train_test(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}")
    
    generations = 50
    population_count = 50
    sample_size = 5
    cross_breed_prob = 0.5
    add_child_prob = 0.4
    
    print("Starting evolution...")
    start_time = time.time()
    
    best_tree = evolution(
        generations=generations,
        population_count=population_count,
        train_X=train_X,
        train_Y=train_Y,
        sample_size=sample_size,
        cross_breed_prob=cross_breed_prob,
        add_child_prob=add_child_prob
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Evolution finished in {duration:.2f} seconds.")
    
    if best_tree:
        test_accuracy = calculate_accuracy(best_tree, test_X, test_Y)
        train_accuracy = calculate_accuracy(best_tree, train_X, train_Y)
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:  {test_accuracy:.4f}")
    else:
        print("Evolution failed to produce a valid tree.")

def main():
    datasets = [
        "breast_cancer",
        "winequality_red",
        "winequality_white",
        "airline_passenger_satisfaction"
    ]
    
    for ds in datasets:
        run_experiment(ds)

if __name__ == "__main__":
    main()
