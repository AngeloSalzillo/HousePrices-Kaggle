import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Columns to drop completely
TO_DELETE = ["PoolQC", "Alley", "MiscFeature", "Fence", "Utilities"]

# Columns where NaN = "Missing" (Categorical)
MISSING_COLS = [
    "MasVnrType","BsmtQual","BsmtCond","BsmtExposure",
    "BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType",
    "GarageFinish","GarageQual","GarageCond","GarageFinish"
]

# Columns where NaN = 0 (Numerical)
ZERO_COLS = [
    "MasVnrArea","LotFrontage","GarageYrBlt","BsmtHalfBath",
    "BsmtFullBath","TotalBsmtSF","BsmtFinSF1","BsmtFinSF2",
    "GarageCars","GarageArea","BsmtUnfSF"
]

# Columns where NaN = most frequent value (Categorical)
FREQ_COLS = [
    "MSZoning","Exterior1st","Exterior2nd",
    "KitchenQual","SaleType","Electrical"
]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Creates the preprocessing pipeline."""

    # Identify categorical columns not already handled above
    remaining_cat = df.select_dtypes(exclude=["number"]).columns.drop(
        MISSING_COLS + FREQ_COLS
    ).tolist()

    missing_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    frequent_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    zero_pipeline = SimpleImputer(strategy="constant", fill_value=0)
    encode_pipeline = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("freq", frequent_pipeline, FREQ_COLS),
        ("missing", missing_pipeline, MISSING_COLS),
        ("zero", zero_pipeline, ZERO_COLS),
        ("encode_rest", encode_pipeline, remaining_cat)
    ], remainder="passthrough")

    return preprocessor
