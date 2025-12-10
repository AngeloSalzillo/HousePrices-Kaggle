import os
import sys
import pandas as pd
from src.train_xgboost import load_data, train_xgboost

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = "/kaggle/input/house-prices-advanced-regression-techniques"
SAVE_DIR = "results"

X_train, y_train, X_test, ids = load_data(BASE_DIR)

preds = train_xgboost(X_train, y_train, X_test, save_dir=SAVE_DIR)

submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": preds
})

submission.to_csv(f"{SAVE_DIR}/submission.csv", index=False)
print("Submission saved!")
