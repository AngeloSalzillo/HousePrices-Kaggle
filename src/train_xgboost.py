import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, TO_DELETE
from src.features import apply_feature_engineering


def load_data(base_dir: str):
    train = pd.read_csv(os.path.join(base_dir, "train.csv"))
    test = pd.read_csv(os.path.join(base_dir, "test.csv"))

    train = train.dropna(subset=["SalePrice"])
    y = np.log1p(train["SalePrice"])

    X_train = train.drop(["Id", "SalePrice"], axis=1)
    X_test = test.drop(["Id"], axis=1)

    return X_train, y, X_test, test["Id"]


def train_xgboost(X_train, y_train, X_test, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    # Combine for consistent preprocessing
    full = pd.concat([X_train, X_test], axis=0)

    # Drop problematic columns
    full = full.drop(TO_DELETE, axis=1)

    # Feature engineering
    full = apply_feature_engineering(full)

    X_train = full[:len(X_train)]
    X_test = full[len(X_train):]

    # Preprocessor
    preprocessor = build_preprocessor(full)

    # Model pipeline
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", XGBRegressor())
    ])

    # Bayesian hyperparameter search
    search_space = {
        "regressor__n_estimators": Integer(200, 5000),
        "regressor__learning_rate": Real(0.01, 0.1),
        "regressor__max_depth": Integer(2, 8),
        "regressor__colsample_bytree": Real(0.5, 1),
        "regressor__reg_alpha": Real(0, 1),
        "regressor__reg_lambda": Real(0, 1),
        "regressor__subsample": Real(0.5, 1)
    }

    optimizer = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        cv=5,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        verbose=1
    )

    optimizer.fit(X_train, y_train)

    # Save best parameters
    with open(os.path.join(save_dir, "xgb_best_params.json"), "w") as f:
        json.dump(optimizer.best_params_, f, indent=4)

    with open(os.path.join(save_dir, "cv_score.txt"), "w") as f:
        f.write(str(-optimizer.best_score_))

    # Final predictions
    preds = np.expm1(optimizer.predict(X_test))
    return preds
