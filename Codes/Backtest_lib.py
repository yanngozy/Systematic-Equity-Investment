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




def train_lgbm(train_df, feature_cols, cat_cols_tree, train_size=0.9, task_type='CPU', algo='lgbm'):
    
    n_train = int(len(train_df) * train_size)
    X_train = train_df[feature_cols].iloc[:n_train]
    X_val   = train_df[feature_cols].iloc[n_train:]
    y_train = train_df["target"].iloc[:n_train]
    y_val   = train_df["target"].iloc[n_train:]
    
    cat_feature_indices = [feature_cols.index(c) for c in cat_cols_tree]

    if algo == 'lgbm':
        from lightgbm import LGBMRegressor
        import lightgbm as lgb
        model = LGBMRegressor(
            objective="regression", metric="rmse", n_estimators=2000,
            learning_rate=0.03, num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            categorical_feature=cat_cols_tree,
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False), lgb.log_evaluation(period=0)]
        )
    elif algo == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE", iterations=2000,
            learning_rate=0.01, depth=6, l2_leaf_reg=5.0, bagging_temperature=0.65,
            min_data_in_leaf=80, random_strength=0.8, random_seed=42,
            task_type=task_type, verbose=False, allow_writing_files=False
        )
        model.fit(X_train, y_train, 
                  eval_set=(X_val, y_val), 
                  cat_features=cat_feature_indices
                 )

    return model

def generate_score(data, feature_cols, train_size=0.8, algo='lgbm', start_year=2017, end_year=2026):
    results = []
    
    data = data.copy()
    cat_cols_tree = data[feature_cols].select_dtypes(include=['object', 'category']).columns.to_list()
    for col in cat_cols_tree:
        data[col] = data[col].astype('category')
        
    years = sorted(data["Date"].dt.year.unique())
    task_type = "GPU" if torch.cuda.is_available() else "CPU"

    for year in tqdm.tqdm(years):
        if year < start_year or year > end_year:
            continue
        train_mask = data["Date"].dt.year < year
        test_mask  = data["Date"].dt.year == year
        
        if not train_mask.any() or not test_mask.any():
            continue
            
        train = data[train_mask]
        test  = data[test_mask].copy()
        
        model = train_lgbm(train, feature_cols, cat_cols_tree, train_size=train_size, task_type=task_type, algo=algo)
        test["score"] = model.predict(test[feature_cols])
        
        results.append(test)
        
        del train, test, model
        gc.collect() 

    return pd.concat(results, axis=0)




def metrics_report(data):
    dico = { 
            'maximum drawdown'   :   ( 1 - data['ptf']/ data['ptf'].cummax()).max(),
            'annual return'      :   data['returns'].mean()*252,
            'annual volatility'  :   data['returns'].std()*np.sqrt(252), 
            'turnover'           :    data['turnover_oneway'].mean()      
            
          }

    dico['sharpe'] = dico['annual return']/dico['annual volatility']
    
    return dico
