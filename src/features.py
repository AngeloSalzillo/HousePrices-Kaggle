import pandas as pd
import numpy as np

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to dataset."""

    df = df.copy()


    df["Functional"] = df["Functional"].fillna("Typ")
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalSF"] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["TotalBath"] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df["TotalPorchSF"] = (
        df['WoodDeckSF'] + df['OpenPorchSF'] +
        df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    )
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFlr'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    return df
