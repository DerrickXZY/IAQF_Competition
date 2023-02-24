"""
Prediction on cumulative return from t to t+steps using data on t-1, t-2, ..., 
t-k is put on day t-1
"""

import numpy as np
import pandas as pd
from vecm_utils import VECM, VAR, select_order, select_coint_rank

# change path to the directory where file is
import os
os.chdir(os.path.dirname(__file__))
# # TODO: Change to import Predictor
# import sys
# sys.path.append("../pipeline/")
# from pipeline import Predictor
from typing import List, Tuple, Optional, Mapping
from tqdm import tqdm


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


class BaselinePredictor(Predictor):
    """
    Build VECM / VAR
    """
    def __init__(self):
        super().__init__()

    def train(self, data: pd.DataFrame, deterministic="colo", maxlags=10):
        """
        deterministic: str, "n", "ci", "co", "li", "lo"
        """
        self.train_df_last_row = data.iloc[[-1], :].copy()
        train_df = data.reset_index(drop=True)

        # Choose the best lag order
        # without `.reset_index(drop=True)`: warning - not use date index
        lag_order_info = select_order(data=train_df, maxlags=maxlags, 
                                      deterministic=deterministic)
        # get the best lag order by:
        # lag_order.aic, lag_order.bic, lag_order.fpe, lag_order.hqic
        self.lag_order = lag_order_info.aic
        if self.lag_order == 0:
            self.lag_order = 1

        # Get Cointegration Rank
        # `det_order=-1`: no deterministic terms
        rank_test = select_coint_rank(train_df, det_order=-1, 
            k_ar_diff=self.lag_order, signif=0.05)
        self.coint_rank = rank_test.rank

        # Decide which model to build
        if self.coint_rank >= 1:
            self.model_type = "VECM"
            vecm = VECM(train_df, 
                        k_ar_diff=self.lag_order, 
                        coint_rank=self.coint_rank,
                        deterministic=deterministic)
            self.model = vecm
            self.model_result = vecm.fit()
            self.k_ar = vecm.k_ar
            # prepare data for prediction
            self.last_observations = train_df.iloc[-self.k_ar:].values
        else:
            self.model_type = "VAR"
            train_rtn_df = train_df.pct_change().dropna().reset_index(drop=True)
            train_rtn_df.columns = [x + "_rtn" for x in data.columns]
            var = VAR(train_rtn_df)
            # VAR model chooses best lag order according to `ic` implicitly
            lag_order_info = var.select_order(maxlags=10)
            if lag_order_info.aic == 0:
                self.model_result = var.fit() # implies lags = 1
            else:
                self.model_result = var.fit(maxlags=maxlags, ic="aic")
            self.model = var
            # prepare data for prediction
            self.k_ar = self.model_result.k_ar
            self.lag_order = self.k_ar
            self.last_observations = train_rtn_df.iloc[-self.k_ar:].values

    def predict(self, data: pd.DataFrame, steps: int, in_sample: bool = False) \
        -> pd.Series:
        """
        Day by day forecast
        """
        if in_sample:
            # In-Sample Prediction
            
            if self.model_type == "VECM":
                pred_list = []
                nd = data.values
                pred_len = data.shape[0] - self.k_ar + 1
                for i in range(pred_len):
                    pred_during_steps = self.model_result.predict(
                        steps=steps, last_observations=nd[i: i + self.k_ar]
                    )
                    pred_at_step = pred_during_steps[[-1], :]
                    pred_list.append(pred_at_step)
                price_pred = np.vstack(pred_list)

            elif self.model_type == "VAR":
                rtn_pred_list = []
                train_rtn_df = data.pct_change().dropna()
                rtn_nd = train_rtn_df.values
                pred_len = train_rtn_df.shape[0] - self.k_ar + 1
                for i in range(pred_len):
                    rtn_pred_during_steps = self.model_result.forecast(
                        rtn_nd[i: i + self.k_ar], steps=steps
                    )
                    rtn_pred_at_step = (1 + rtn_pred_during_steps).prod(axis=0, 
                        keepdims=True) - 1
                    rtn_pred_list.append(rtn_pred_at_step)
                rtn_pred = np.vstack(rtn_pred_list)
                price_pred = (1 + rtn_pred) * data.values[-pred_len:]

            # prediction using t-1, t-2, ..., t-k is put on day t-1
            price_pred_df = pd.DataFrame(price_pred, 
                                         index=train_df.index[-pred_len:], 
                                         columns=train_df.columns)
            # calculate predicted return spread
            rtn_pred_df = price_pred_df / train_df.iloc[-pred_len:, :] - 1
            rtn_spread_pred_s = rtn_pred_df.iloc[:, 0] - rtn_pred_df.iloc[:, 1]
            return rtn_spread_pred_s

        else:
            # Out-Of-Sample Prediction
            concat_df = pd.concat([self.train_df_last_row, data])
            pred_len = concat_df.shape[0]
            if self.model_type == "VECM":
                pred_list = []
                nd = np.vstack([self.last_observations, data.values])
                for i in range(pred_len):
                    pred_during_steps = self.model_result.predict(
                        steps=steps, last_observations=nd[i: i + self.k_ar]
                    )
                    pred_at_step = pred_during_steps[[-1], :]
                    pred_list.append(pred_at_step)
                price_pred = np.vstack(pred_list)

            elif self.model_type == "VAR":
                rtn_pred_list = []
                test_rtn_df = concat_df.pct_change().dropna()
                rtn_nd = np.vstack([self.last_observations, test_rtn_df.values])
                for i in range(pred_len):
                    rtn_pred_during_steps = self.model_result.forecast(
                        rtn_nd[i: i + self.k_ar], steps=steps
                    )
                    rtn_pred_at_step = (1 + rtn_pred_during_steps).prod(axis=0, 
                        keepdims=True) - 1
                    rtn_pred_list.append(rtn_pred_at_step)
                rtn_pred = np.vstack(rtn_pred_list)
                price_pred = (1 + rtn_pred) * concat_df.values

            # prediction using t-1, t-2, ..., t-k is put on day t-1
            price_pred_df = pd.DataFrame(price_pred, index=concat_df.index, 
                                        columns=concat_df.columns)
            # calculate predicted return spread
            rtn_pred_df = price_pred_df / concat_df - 1
            rtn_spread_pred_s = rtn_pred_df.iloc[:, 0] - rtn_pred_df.iloc[:, 1]
            return rtn_spread_pred_s

    def periodic_train_predict(self, data: pd.DataFrame, split_date: str, \
        steps: int) -> pd.Series:
        split_year = int(split_date.split("-")[0])
        output_series_list = []
        while split_year <= data.index[-1].year:
            split_date = f"{split_year}-01-01"
            test_end_date = f"{split_year + 1}-01-01"
            train_df = data[data.index < split_date].copy()
            test_df = data[
                (data.index >= split_date) & (data.index < test_end_date)
            ].copy()
            self.train(train_df)
            rtn_spread_pred_s = self.predict(test_df, steps)
            output_series_list.append(rtn_spread_pred_s)
            split_year += 1
        output_series = pd.concat(output_series_list)
        # `keep="first"`: for the last day in the training period, the 
        # prediction comes from the old model, but not the new one. for
        # example, at 2017-12-31, we update the model, but the prediction 
        # on next day still comes from the old model
        output_series = \
            output_series[~output_series.index.duplicated(keep="first")]
        return output_series

if __name__ == "__main__":
    split_date = "2017-01-01"
    freq_mapping = dict(zip(["d", "w", "M"], [1, 5, 21]))
    modes = ["predict", "periodic_train_predict"]
    
    # Set Path
    data_path = "../data/"
    pred_path = f"../prediction/baseline/"
    param_path = f"../parameter/baseline/"
    for mode in modes:
        if not os.path.exists(f"{pred_path}{mode}/"):
            os.makedirs(f"{pred_path}{mode}/")
        if not os.path.exists(f"{param_path}{mode}/"):
            os.makedirs(f"{param_path}{mode}/")

    # Load Data
    raw_price_df = pd.read_csv(data_path + "price_df.csv", 
                               parse_dates=["Date"])
    raw_price_df.sort_values(["ETF_Ticker", "Date"], inplace=True)
    price_s = raw_price_df.set_index(["ETF_Ticker", "Date"]).squeeze()
    feature_df = pd.read_csv(data_path + "TrainingSet.csv")
    pair_arr = np.unique(feature_df["Ticker_Pair"].values)
    
    # Make predictions on each pair under different modes and frequencies
    for freq, steps in tqdm(freq_mapping.items()):
        for mode in tqdm(modes):
            output_list = []
            estimated_parameters = []
            for pair in tqdm(pair_arr):
                pair_list = pair.split("_")
                pair_s = price_s.loc[pair_list, :].copy()
                pair_df = pair_s.unstack("ETF_Ticker")
                pair_df.dropna(inplace=True)

                # Train test split
                train_df = pair_df.loc[pair_df.index < split_date].copy()
                test_df = pair_df.loc[pair_df.index >= split_date].copy()

                # Build Predictor
                predictor = BaselinePredictor()
                if mode == "predict":
                    predictor.train(train_df)
                    rtn_spread_pred_out_of_sample_s = \
                        predictor.predict(test_df, steps)
                    # get in-sample prediction
                    rtn_spread_pred_in_sample_s = \
                        predictor.predict(train_df, steps, True)

                elif mode == "periodic_train_predict":
                    rtn_spread_pred_out_of_sample_s = \
                        predictor.periodic_train_predict(
                            pair_df, split_date, steps
                        )
                    # get in-sample prediction
                    rtn_spread_pred_in_sample_s = \
                        predictor.predict(train_df, steps, True)
                    
                # get estimated parameters
                parameter_dict = {"model_type": predictor.model_type}
                result = predictor.model_result
                if predictor.model_type == "VECM":
                    # see parameter explanations:
                    # https://www.statsmodels.org/dev/generated/statsmodels.
                    # tsa.vector_ar.vecm.VECMResults.html
                    # model representation:
                    # https://www.statsmodels.org/dev/generated/statsmodels.
                    # tsa.vector_ar.vecm.VECM.html
                    parameter_dict["coint_rank"] = result.coint_rank
                    # k_ar: number of lags in the VAR representation
                    # k_ar_diff: number of lags in VECM
                    parameter_dict["k_ar_diff"] = result.k_ar - 1
                    # alpha: coefficients of error correction terms(s)
                    parameter_dict["alpha"] = result.alpha
                    # beta: coefficients to build error correction term(s)
                    parameter_dict["beta"] = result.beta
                    # gamma: coefficients of lag terms
                    parameter_dict["gamma"] = result.gamma
                    # det_coef: coefficients of deterministic terms,
                    # i.e. constant and trend term here
                    parameter_dict["det_coef"] = result.det_coef
                elif predictor.model_type == "VAR":
                    # see parameter explanations:
                    # https://www.statsmodels.org/dev/generated/statsmodels.
                    # tsa.vector_ar.var_model.VARResults.html
                    # k_ar: number of lags
                    parameter_dict["k_ar"] = result.k_ar
                    # params: all parameters
                    parameter_dict["params"] = result.params

                estimated_parameters.append(parameter_dict)

                # aggregate in-sample and out-of-sample prediction
                rtn_spread_pred_s = pd.concat([
                    rtn_spread_pred_in_sample_s.iloc[:-1], 
                    rtn_spread_pred_out_of_sample_s
                ])

                # calculate actual return spread
                # rtn_spread_actual_out_of_sample_df = pd.concat(
                #     [train_df.iloc[[-1], :], test_df]
                # ).pct_change(steps).shift(-steps).dropna()
                rtn_spread_actual_df = \
                    pair_df.pct_change(steps).shift(-steps).dropna()
                rtn_spread_actual_s = rtn_spread_actual_df.iloc[:, 0] - \
                    rtn_spread_actual_df.iloc[:, 1]
        
                output_i_df = pd.concat(
                    [rtn_spread_pred_s, rtn_spread_actual_s],
                    axis=1,
                    join="inner"
                )
                output_i_df.columns = ["pred_spread", "actual_spread"]

                # Reformat
                output_i_df.index.name = "Date"
                output_i_df.reset_index(inplace=True)
                output_i_df["pair"] = pair
                output_i_df = output_i_df.reindex(
                    np.roll(output_i_df.columns.values, 1), axis=1
                )
                output_list.append(output_i_df)
                
            output_df = pd.concat(output_list, ignore_index=True)
            # print(output_df)
            output_df.to_pickle(
                f"{pred_path}{mode}/ReturnSpreadPredictions_{freq}.pkl"
            )

            parameter_df = pd.DataFrame(estimated_parameters, index=pair_arr)
            parameter_df.to_csv(f"{param_path}{mode}/parameters_{freq}.csv")
