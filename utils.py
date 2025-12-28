import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(path : str, delimiter=','):
    fh = open(path, 'r')
    return pd.read_csv(fh, delimiter=delimiter)

def load_data(path : str):
    fh = open(path, 'r')
    return pd.read_csv(path, names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])


def split_train_test(name : str):
    """"
        valid names are: 
            airline_passenger_satisfaction,
            breast_cancer,
            winequality_white,
            winequality_red

        use: returns train_X, test_X, train_Y, test_Y
        everything is a pd.DataFrame
    """
    df = None
    if name == "breast_cancer":
        path = "data/breast_cancer/breast-cancer.data"
        df = load_data(path)
    else:
        if name == "airline_passenger_satisfaction":
            path = "data/airline-passenger-satisfaction"
            train = load_csv(path + "/train.csv")
            test = load_csv(path + "/test.csv")
            train_X, train_Y = split_data_target(train, name)
            test_X, test_Y = split_data_target(test, name)
            return train_X, test_X, train_Y, test_Y
        if name == "winequality_red":
            path = "data/wine_quality/winequality-red.csv"
        elif name == "winequality_white":
            path = "data/wine_quality/winequality-white.csv"
        else:
            raise ValueError("wrong name")
        df = load_csv(path, ';')
    train, test = train_test_split(df, test_size=0.2)
    train_X, train_Y = split_data_target(train, name)
    test_X, test_Y = split_data_target(test, name)
    return train_X, test_X, train_Y, test_Y



def split_data_target(df : pd.DataFrame, name: str):
    targets = {
        'airline_passenger_satisfaction': 'satisfaction',
        'winequality_red': 'quality',
        'winequality_white': 'quality',
        'breast_cancer': 'irradiat'
    }
    target = df[targets[name]]
    data = df.drop(targets[name], axis=1)
    return data, target

