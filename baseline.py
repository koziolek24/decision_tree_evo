from utils import split_train_test
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

datasets = [
    'winequality_red',
    'winequality_white',
    'breast_cancer',
    'airline_passenger_satisfaction'
]


configs = {
    'breast_cancer': {
        'n_estimators': 20,
        'learning_rate': 0.1,
        'max_depth': 5,
        'eval_metric': 'mlogloss',
        'enable_categorical': True
    },
    'winequality_red': {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 8,
        'eval_metric': 'mlogloss',
        'enable_categorical': True
    },
    'winequality_white': {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 8,
        'eval_metric': 'mlogloss',
        'enable_categorical': True
    },
    'airline_passenger_satisfaction': {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 8,
        'eval_metric': 'mlogloss',
        'enable_categorical': True
    }
}

for dataset in datasets:
    X_train, X_test, Y_train, Y_test = split_train_test(dataset)
    
    for col in X_train.select_dtypes(['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)

    config = configs[dataset]
    model = xgb.XGBClassifier(**config)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"{accuracy*100:.2f}% for {dataset}")
