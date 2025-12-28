import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(path : str):
    fh = open(path, 'r')
    return pd.read_csv(fh)

def load_data(path : str):
    fh = open(path, 'r')
    return pd.read_csv(path, names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])


def split_train_test_csv(name : str):
    df = None
    if name == "breast_cancer":
        path = "data/breast_cancer/breast-cancer.data"
        df = load_data(path)
    else:
        if name == "airline_passenger_satisfaction":
            path = "data/airline-passenger-satisfaction"
            train = load_csv(path + "/train.csv")
            test = load_csv(path + "/test.csv")
            return train, test
        if name == "winequality_red":
            path = "data/wine_quality/winequality-red.csv"
        elif name == "winequality_white":
            path = "data/wine_quality/winequality-white.csv"
        else:
            raise ValueError("wrong name")
        df = load_csv(path)
    return train_test_split(df, test_size=0.2)
