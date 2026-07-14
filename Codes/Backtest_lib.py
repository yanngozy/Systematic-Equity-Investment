import numpy as np 
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeClassifier,Ridge
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor




def train_lgbm(train_df,feature_cols,algo='lgbm'):

    X = train_df[feature_cols]
    
    y = train_df["target"]

    if algo=='lgbm':
        model = LGBMRegressor(
                                objective="regression",
                                n_estimators=500,
                                learning_rate=0.03,
                                num_leaves=64,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42
                            )
    elif algo=='catboost':
        model = CatBoostRegressor(
                                    loss_function="RMSE",       
                                    eval_metric="RMSE",
                                    iterations=500,
                                    learning_rate=0.01,
                                    depth=6,
                                    l2_leaf_reg=5.0,
                                    bagging_temperature=0.65,
                                    min_data_in_leaf=80,
                                    random_strength=0.8,
                                    random_seed=42,
                                    verbose=False,
                                    allow_writing_files=False,
                                )
    
    model.fit(X, y)

    return model

def generate_score(data, feature_cols, algo='lgbm', start_year=2017, end_year=2020):

    results = []

    years = sorted(data["Date"].dt.year.unique())

    for year in years:

        if year < start_year or year > end_year:
            continue

        train = data[data["Date"].dt.year < year]

        test = data[data["Date"].dt.year == year].copy()

        model = train_lgbm(train,feature_cols,algo)

        test["score"] = model.predict(test[feature_cols])

        results.append(test)

    return pd.concat(results)



def metrics_report(data):
    dico = { 
            'maximum drawdown'   :   ( 1 - data['ptf']/ data['ptf'].cummax()).max(),
            'annual return'      :   data['returns'].mean()*252,
            'annual volatility'  :   data['returns'].std()*np.sqrt(252), 
            'turnover'           :    data['turnover_oneway'].mean()      
            
          }

    dico['sharpe'] = dico['annual return']/dico['annual volatility']
    
    return dico