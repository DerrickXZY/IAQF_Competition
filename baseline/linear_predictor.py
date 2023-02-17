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


class LinearPredictor(Predictor):
    """
    Build VECM / VAR
    """
    def __init__(self):
        super().__init__()

    def train(self, data: pd.DataFrame, deterministic="colo", maxlags=10):
        """
        deterministic: str, "n", "ci", "co", "li", "lo"
        """
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
            train_rtn_df.columns = [x + "_rtn" for x in pair_df.columns]
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
        
        self.train_df_last_row = train_df.iloc[[-1], :].copy()

    def predict(self, data: pd.DataFrame, save_path: Optional[str] = None) \
        -> pd.DataFrame:
        """
        Day by day forecast
        """
        if self.model_type == "VECM":
            pred_list = []
            nd = np.vstack([self.last_observations, data.values])
            for i in range(data.shape[0]):
                pred_i = self.model_result.predict(
                    steps=1, last_observations=nd[i: i + self.k_ar]
                )
                pred_list.append(pred_i)
            pred = np.vstack(pred_list)

        elif self.model_type == "VAR":
            rtn_pred_list = []
            test_df = pd.concat([self.train_df_last_row, data], axis=0,
                                ignore_index=True)
            test_rtn_df = test_df.pct_change().dropna()
            rtn_nd = np.vstack([self.last_observations, test_rtn_df.values])
            for i in range(test_rtn_df.shape[0]):
                rtn_pred_i = self.model_result.forecast(
                    rtn_nd[i: i + self.k_ar], steps=1
                )
                rtn_pred_list.append(rtn_pred_i)
            rtn_pred = np.vstack(rtn_pred_list)
            pred = (1 + rtn_pred).cumprod(axis=0) * \
                self.train_df_last_row.values

        pred_df = pd.DataFrame(pred, index=data.index, columns=data.columns)
        # Rearrange output format
        # calculate predicted return spread
        rtn_pred_df = pd.concat(
            [self.train_df_last_row, pred_df]
        ).pct_change().dropna()
        rtn_spread_pred_s = rtn_pred_df.iloc[:, 0] - rtn_pred_df.iloc[:, 1]
        # calculate actual return spread
        rtn_actual_df = pd.concat(
            [self.train_df_last_row, data]
        ).pct_change().dropna()
        rtn_spread_actual_s = \
            rtn_actual_df.iloc[:, 0] - rtn_actual_df.iloc[:, 1]
        output_df = pd.concat([rtn_spread_pred_s, rtn_spread_actual_s], axis=1)
        output_df.columns = ["pred_spread", "actual_spread"]
        if save_path:
            output_df.to_pickle(save_path)
        return output_df

    def periodic_train_predict(self, data: pd.DataFrame, \
        save_path: Optional[str] = None) -> pd.DataFrame:
        split_year = 2017
        output_dfs = []
        while split_year <= data.index[-1].year:
            split_date = f"{split_year}-01-01"
            test_end_date = f"{split_year + 1}-01-01"
            train_df = data[data.index < split_date].copy()
            test_df = data[
                (data.index >= split_date) & (data.index < test_end_date)
            ].copy()
            self.train(train_df)
            output_df_i = self.predict(test_df)
            output_dfs.append(output_df_i)
            split_year += 1
        output_df = pd.concat(output_dfs)
        if save_path:
            output_df.to_pickle(save_path)
        return output_df

if __name__ == "__main__":
    freqs = ["d", "w", "M"]
    modes = ["predict", "periodic_train_predict"]
    
    # Set Path
    data_path = "../data/"
    save_path = f"../prediction/linear_model/"
    for mode in modes:
        if not os.path.exists(f"{save_path}{mode}/"):
            os.makedirs(f"{save_path}{mode}/")

    # Load Pairs
    raw_price_df = pd.read_csv(data_path + "price_df.csv", parse_dates=["Date"])
    raw_price_df.sort_values(["ETF_Ticker", "Date"], inplace=True)
    feature_df = pd.read_csv(data_path + "TrainingSet.csv")
    pair_arr = np.unique(feature_df["Ticker_Pair"].values)
    
    # Make predictions on each pair under 
    for freq in tqdm(freqs):
        if freq == "d":
            price_s = raw_price_df.set_index(["ETF_Ticker", "Date"]).squeeze()
        else:
            price_s = raw_price_df.groupby("ETF_Ticker").apply(lambda df: 
                df.set_index("Date")["ETF Price"].resample(freq).last())
        for mode in tqdm(modes):
            output_list = []
            for pair in pair_arr:
                pair_list = pair.split("_")
                pair_s = price_s.loc[pair_list, :].copy()
                pair_df = pair_s.unstack("ETF_Ticker")
                pair_df.dropna(inplace=True)

                # Build Predictor
                predictor = LinearPredictor()
                if mode == "predict":
                    # train test split
                    train_df = pair_df.loc[pair_df.index < "2017-01-01"].copy()
                    test_df = pair_df.loc[pair_df.index >= "2017-01-01"].copy()
                    predictor.train(train_df)
                    output_i_df = predictor.predict(
                        test_df#, f"{save_path}{pair}_{freq}.pkl"
                    )
                elif mode == "periodic_train_predict":
                    output_i_df = predictor.periodic_train_predict(
                        pair_df#, f"{save_path}{pair}_{freq}.pkl"
                    )
                # print(output_i_df.head())
                # print()
                # print(output_i_df.tail())
                output_i_df.index.name = "Date"
                output_i_df.reset_index(inplace=True)
                output_i_df["pair"] = pair
                output_i_df = output_i_df.reindex(
                    ["pair"] + list(output_i_df.columns[:-1]), axis=1
                )
                output_list.append(output_i_df)
                
            output_df = pd.concat(output_list)
            output_df.to_pickle(
                f"{save_path}{mode}/ReturnSpreadPredictions_{freq}.pkl"
            )
