from typing import List, Tuple, Optional, Mapping
import pickle
import pandas as pd

class Predictor():
    def __init__(self):
        pass

    def train(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None):
        pass

    def predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        pass

    def periodic_train_predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        pass


class MockPredictor(Predictor):
    def __init__(self):
        pass

    def train(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None):
        pass

    def predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        return pd.read_csv("mock_evaluation_df.csv")

    def periodic_train_predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        return self.predict(data, params)


class LinearPredictor(Predictor):
    def __init__(self):
        pass

    def train(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None):
        pass

    def predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        if params is None:
            params = {
                'pred_period': 'D'
            }
        pred_period = params['pred_period']
        if pred_period == 'D':
            with open('../prediction/baseline/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'M':
            with open('../prediction/baseline/predict/ReturnSpreadPredictions_M.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'W':
            with open('../prediction/baseline/predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
                return pickle.load(f)
        with open('../prediction/baseline/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
            return pickle.load(f)

    def periodic_train_predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        if params is None:
            params = {
                'pred_period': 'D'
            }
        pred_period = params['pred_period']
        if pred_period == 'D':
            with open('../prediction/baseline/periodic_train_predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'M':
            with open('../prediction/baseline/periodic_train_predict/ReturnSpreadPredictions_M.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'W':
            with open('../prediction/baseline/periodic_train_predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
                return pickle.load(f)
        with open('../prediction/baseline/periodic_train_predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
            return pickle.load(f)


class ElasticNetPredictor(Predictor):
    def __init__(self):
        pass

    def train(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None):
        pass

    def predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        if params is None:
            params = {
                'pred_period': 'D'
            }
        pred_period = params['pred_period']
        if pred_period == 'D':
            with open('../prediction/elastic_net/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'M':
            with open('../prediction/elastic_net/predict/ReturnSpreadPredictions_M.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'W':
            with open('../prediction/elastic_net/predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
                return pickle.load(f)
        with open('../prediction/elastic_net/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
            return pickle.load(f)

    def periodic_train_predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        if params is None:
            params = {
                'pred_period': 'D'
            }
        pred_period = params['pred_period']
        if pred_period == 'D':
            with open('../prediction/elastic_net/periodic_train_predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'M':
            with open('../prediction/elastic_net/periodic_train_predict/ReturnSpreadPredictions_M.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'W':
            with open('../prediction/elastic_net/periodic_train_predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
                return pickle.load(f)
        with open('../prediction/elastic_net/periodic_train_predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
            return pickle.load(f)


class XGBPredictor(Predictor):
    def __init__(self):
        pass

    def train(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None):
        pass

    def predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        # return pd.read_csv("mock_evaluation_df.csv")
        if params is None:
            params = {
                'pred_period': 'D'
            }
        pred_period = params['pred_period']
        if pred_period == 'D':
            with open('../prediction/xgboost_model/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'M':
            with open('../prediction/xgboost_model/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
                return pickle.load(f)
        if pred_period == 'W':
            with open('../prediction/xgboost_model/predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
                return pickle.load(f)

    def periodic_train_predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        return self.predict(data, params)


class LSTMPredictor(Predictor):
    def __init__(self):
        pass

    def train(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None):
        pass

    def predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        if params is None:
            params = {
                'pred_period': 'D'
            }
        pred_period = params['pred_period']
        # if pred_period == 'D':
        #     with open('../prediction/lstm/predict/ReturnSpreadPredictions_d.pkl', 'rb') as f:
        #         return pickle.load(f)
        # if pred_period == 'M':
        #     with open('../prediction/lstm/predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
        #         return pickle.load(f)
        if pred_period == 'W':
            with open('../prediction/lstm/predict/ReturnSpreadPredictions_w.pkl', 'rb') as f:
                lstm_pred_df = pickle.load(f)
                lstm_pred_df.columns = lstm_pred_df.columns.map(lambda x: x[0])
                return lstm_pred_df

    def periodic_train_predict(self, data: Optional[pd.DataFrame] = None, params: Optional[Mapping] = None) -> pd.DataFrame:
        return self.predict(data, params)
