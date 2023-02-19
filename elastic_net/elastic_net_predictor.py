"""
Prediction on cumulative return from t to t+steps using data on t-1, t-2, ..., 
t-k is put on day t-1
"""

import numpy as np
import pandas as pd

# change path to the directory where file is
import os
os.chdir(os.path.dirname(__file__))
# # TODO: Change to import Predictor
# import sys
# sys.path.append("../pipeline/")
# from pipeline import Predictor
from typing import List, Tuple, Optional, Mapping
from tqdm import tqdm

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# TODO: Change to import Predictor
class Predictor():
    def __init__(self):
        pass
    
    def train(self, data: pd.DataFrame, params: Optional[Mapping] = None):
        pass

    def predict(self, data: pd.DataFrame, params: Optional[Mapping] = None) -> pd.DataFrame:
        pass

    def periodic_train_predict(self, data: pd.DataFrame, params: Optional[Mapping] = None) -> pd.DataFrame:
        pass


class ElasticNetPredictor(Predictor):
    """
    Build VECM / VAR
    """
    def __init__(self):
        super().__init__()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              param_grid: dict):
        elastic_net = ElasticNet()
        tss = TimeSeriesSplit(n_splits=5)
        cv = GridSearchCV(elastic_net, param_grid, cv=tss, n_jobs=-1)
        cv.fit(X_train, y_train)
        self.model = cv.best_estimator_
        self.cv_best_score = cv.best_score_
        self.best_params = cv.best_params_

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X_test)
        return pd.Series(y_pred, index=X_test.index)

    def periodic_train_predict(self, X: pd.DataFrame, y: pd.DataFrame, 
                               split_date: str, param_grid: dict) -> pd.Series:
        split_year = int(split_date.split("-")[0])
        output_series_list = []
        while split_year <= y.index[-1].year:
            split_date = f"{split_year}-01-01"
            test_end_date = f"{split_year + 1}-01-01"
            X_train = X[X.index < split_date]
            y_train = y[y.index < split_date]
            X_test = X[(X.index >= split_date) & (X.index < test_end_date)]
            self.train(X_train, y_train, param_grid)
            y_pred_series = self.predict(X_test)
            output_series_list.append(y_pred_series)
            split_year += 1
        output_series = pd.concat(output_series_list)
        return output_series

if __name__ == "__main__":
    split_date = "2018-01-01"
    freq_mapping = dict(zip(["d", "w", "M"], [1, 5, 21]))
    modes = ["predict", "periodic_train_predict"]
    
    # Set Path
    data_path = "../data/"
    save_path = f"../prediction/elastic_net/"
    for mode in modes:
        if not os.path.exists(f"{save_path}{mode}/"):
            os.makedirs(f"{save_path}{mode}/")

    # Load Data
    df = pd.read_csv(data_path + "Final TrainingSet.csv", 
                     parse_dates=["Date"],
                     index_col=0)
    pair_arr = np.unique(df["Ticker_Pair"].values)
    df.set_index(["Ticker_Pair", "Date"], inplace=True)
    df.sort_index(inplace=True)
    common_features = [feature for feature in df.columns 
                       if not feature.startswith("Y_")]
    target_prefix = "Y_Fwd_Total_Ret_Pct_"
    param_grid = {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    
    # Make predictions on each pair under different modes and frequencies
    for freq, steps in tqdm(freq_mapping.items()):
        for mode in tqdm(modes):
            output_list = []
            for pair in tqdm(pair_arr):
                target = f"{target_prefix}{steps}"
                ar_features = [feature for feature in df.columns 
                               if feature.startswith(f"Y_{steps}")]
                features = common_features + ar_features
                pair_df = df.loc[pair, features + [target]].dropna()

                # Train test split
                X_train = pair_df.loc[pair_df.index < split_date, features]
                y_train = pair_df.loc[pair_df.index < split_date, target]
                X_test = pair_df.loc[pair_df.index >= split_date, features]
                y_test = pair_df.loc[pair_df.index >= split_date, target]

                # Build Predictor
                predictor = ElasticNetPredictor()
                if mode == "predict":
                    predictor.train(X_train, y_train, param_grid)
                    y_pred = predictor.predict(X_test)
                elif mode == "periodic_train_predict":
                    X = pair_df[features]
                    y = pair_df[target]
                    y_pred = predictor.periodic_train_predict(
                        X, y, split_date, param_grid
                    )

                # get actual return spread from y_test
                output_i_df = pd.concat([y_pred, y_test], axis=1)
                output_i_df.columns = ["pred_spread", "actual_spread"]

                # Reformat
                output_i_df.index.name = "Date"
                output_i_df.reset_index(inplace=True)
                output_i_df["pair"] = pair
                output_i_df = output_i_df.reindex(
                    np.roll(output_i_df.columns.values, 1), axis=1
                )
                output_list.append(output_i_df)
                
            output_df = pd.concat(output_list)
            # print(output_df)
            output_df.to_pickle(
                f"{save_path}{mode}/ReturnSpreadPredictions_{freq}.pkl"
            )
