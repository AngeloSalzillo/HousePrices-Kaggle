## **House Prices Regression – Bayesian Hyperparameter Optimization**

This repository contains experiments on predicting house prices using machine learning, with a focus on Random Forest and XGBoost models,
combined with Bayesian hyperparameter optimization. The goal is to achieve accurate predictions for the Kaggle House Prices: Advanced 
Regression Techniques competition.  




### **Data**

Download the dataset from Kaggle: House Prices - Advanced Regression Techniques
Place the files under /kaggle/input/house-prices-advanced-regression-techniques.
- train.csv – Training data with target variable SalePrice
- test.csv – Test data for predictions
  



### **Features Engineered**

1. HouseAge – Age of the house at the time of sale
2. TotalSF – Total square footage (basement + 1st floor + 2nd floor)
3. TotalBath – Total number of bathrooms (full + half)
4. TotalPorchSF – Total porch area
5. RemodAge – Years since last remodeling
6. Binary flags: HasPool, Has2ndFlr, HasGarage, HasBsmt, HasFireplace



  
### **Preprocessing**

- Drop columns with too many missing values: PoolQC, Alley, MiscFeature, Fence, Utilities
- Fill NaN for categorical features with "Missing" or most frequent value
- Fill NaN for numerical features with 0
- One-hot encode categorical features
- Combine feature engineering with preprocessing in a pipeline


  

### **Final Result**

The model the resulted to yield the best prediction in terms of accuracy was XGBoost with 
hyperparameter tuning handled with Bayes Search. 
Its best parameters are stored in the "results" folder, and allowed to achieve a score of 0.12 (LRMSE).
The other non winning models are stored in the folder "previous trials"
