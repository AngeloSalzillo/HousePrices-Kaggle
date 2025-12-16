import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.train_xgboost import load_data, train_xgboost

BASE_DIR = "/kaggle/input/house-prices-advanced-regression-techniques"
SAVE_DIR = "/kaggle/working"

X_train, y_train, X_test, ids = load_data(BASE_DIR)

preds = train_xgboost(X_train, y_train, X_test, save_dir=SAVE_DIR)

submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": preds
})

submission.to_csv(f"{SAVE_DIR}/submission.csv", index=False)
print("Submission saved!")
