from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
from evolution import evolution

class EvoTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        population_count=200,
        sample_size=5,
        cross_breed_prob=0.8,
        add_child_prob=0.6,
        patience=10,
        penalty_rate=0.001,
        elitism_rate=0.1,
        penalty_type='linear'
    ):
        self.population_count = population_count
        self.population_count = population_count
        self.sample_size = min(sample_size, population_count)
        self.cross_breed_prob = cross_breed_prob
        self.add_child_prob = add_child_prob
        self.patience = patience
        self.penalty_rate = penalty_rate
        self.elitism_rate = elitism_rate
        self.penalty_type = penalty_type

    def fit(self, X, y):
        
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = unique_labels(y_arr)
        
        X_df = pd.DataFrame(X_arr)
        y_series = pd.Series(y_arr)
        
        self.tree_ = evolution(
            population_count=self.population_count,
            train_X=X_df,
            train_Y=y_series,
            sample_size=self.sample_size,
            cross_breed_prob=self.cross_breed_prob,
            add_child_prob=self.add_child_prob,
            patience=self.patience,
            penalty_rate=self.penalty_rate,
            elitism_rate=self.elitism_rate,
            penalty_type=self.penalty_type
        )
        
        return self

    def predict(self, X):
        check_is_fitted(self)
        try:
            X = check_array(X)
        except ValueError:
            X = np.nan_to_num(X, nan=0.0)
            X = check_array(X)
        return self.tree_.predict(X)
