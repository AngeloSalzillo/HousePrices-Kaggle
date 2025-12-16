# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#import libraries and classes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

#load input data
dirname = "/kaggle/input/house-prices-advanced-regression-techniques"
train_set = pd.read_csv(os.path.join(dirname, "train.csv"))
final_test_set = pd.read_csv(os.path.join(dirname,"test.csv"))




#EDA

pd.set_option("display.max_rows", None) #display all rows of a data frame
train_set = train_set.dropna(subset=["SalePrice"]) #deleting rows witout SalePrice value
X_train = train_set.drop(["Id","SalePrice"], axis=1)
X_test = final_test_set.drop(["Id"], axis=1)
y_train = train_set["SalePrice"]
num_train = X_train.shape[0]

#plt.figure() #skew check
#sns.histplot(y, kde=True) 
#plt.show()

y_train = np.log1p(y_train) #skew fix
#plt.figure()
#sns.histplot(y, kde=True)
#plt.show

all_X = pd.concat((X_train,X_test), axis=0)
print(all_X.isna().sum().sort_values(ascending=False))




#PREPROCESSING

#substitution of NaN values and Encoding for categorical features
to_delete_col = ["PoolQC","Alley","MiscFeature","Fence","Utilities"]
missing_col = ["MasVnrType","BsmtQual","BsmtCond","BsmtExposure",
               "BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType","GarageFinish",
              "GarageQual","GarageCond","GarageFinish"]
zero_col = ["MasVnrArea","LotFrontage","GarageYrBlt","BsmtHalfBath",
            "BsmtFullBath","TotalBsmtSF","BsmtFinSF1","BsmtFinSF2","GarageCars","GarageArea","BsmtUnfSF"]
most_frequent_col = ["MSZoning","Exterior1st","Exterior2nd",
                         "KitchenQual","SaleType","Electrical"]
remaining_cat_col = all_X.select_dtypes(exclude = ["number"]).columns.drop(to_delete_col + missing_col + most_frequent_col).to_list()

#NaN values in Functional are filled with Typ as suggested by the data explanation
all_X["Functional"] = all_X["Functional"].fillna("Typ") 

#columns deleted
all_X = all_X.drop(to_delete_col,axis=1) 

#NaN values ---> 0
zero_trans = SimpleImputer(strategy="constant", fill_value=0) 

#NaN values ---> most frequent (categorical) & Encoding
most_frequent_trans = Pipeline(steps = [
    ("NaN filling", SimpleImputer(strategy = "most_frequent")), 
    ("Encoding", OneHotEncoder(handle_unknown = "ignore"))      
])

#NaN values ---> "Missing" & Encoding
missing_trans = Pipeline(steps = [
    ("NaN filling", SimpleImputer(strategy = "constant", fill_value = "Missing")),  
    ("Encoding", OneHotEncoder(handle_unknown = "ignore"))                         
])

encoding_trans = OneHotEncoder(handle_unknown = "ignore")

preprocessor = ColumnTransformer([
    ("most_frequent_transformation", most_frequent_trans, most_frequent_col),
    ("missing_transformation", missing_trans, missing_col),
    ("zero_transformation", zero_trans, zero_col),
    ("encoding", encoding_trans, remaining_cat_col)
], remainder="passthrough")


X_train = all_X[:num_train]
X_test = all_X[num_train:]




#MODEL

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regression", RandomForestRegressor(random_state=1))
])

#hyperparameter tuning
param_grid = {
    "regression__n_estimators": [1000, 2000],
    "regression__max_depth": [None],
    "regression__min_samples_split": [2, 5],
    "regression__min_samples_leaf": [1, 2],
    "regression__max_features": [0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    cv = 5,
    n_jobs = -1,
    scoring = "neg_root_mean_squared_error"
)

grid_search.fit(X_train, y_train)




#PERFORMANCE AND SUBMISSION

print("The best hyperparameters for the Random forest are:\n", grid_search.best_params_)
print("CV error: ", -grid_search.best_score_)

#model applied on the test set
ids = final_test_set["Id"]
final_predictions = np.expm1(grid_search.predict(X_test))

submission = pd.DataFrame({
    "Id" : ids,
    "SalePrice" : final_predictions
})

submission.to_csv("submission.csv", index=False)

submission.head()
