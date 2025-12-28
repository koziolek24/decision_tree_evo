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

for dataset in datasets:
    X_train, X_test, Y_train, Y_test = split_train_test(dataset)
    
    # Minimal preprocessing for XGBoost
    for col in X_train.select_dtypes(['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)

    model = xgb.XGBClassifier(
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 6,
        eval_metric='mlogloss',
        enable_categorical=True
    )
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"{accuracy*100:.2f}% for {dataset}")
