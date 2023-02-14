import numpy as np
import pandas as pd
from vecm_utils import VECM, VAR, select_order, select_coint_rank

from ..pipeline import Predictor # TODO
from typing import List, Tuple, Optional, Mapping

class LinearPredictor(Predictor):
    """
    Build VECM / VAR
    """
    def __init__(self, deterministic="colo"):
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
            self.model_result = vecm.model.fit()
            self.model = vecm
            # prepare data for prediction
            self.last_observations = train_df.iloc[-self.lag_order:].values
        else:
            self.model_type = "VAR"
            train_rtn_df = train_df.pct_change().dropna().reset_index(drop=True)
            train_rtn_df.columns = [x + "_rtn" for x in pair_df.columns]
            var = VAR(train_rtn_df)
            # VAR model chooses best lag order according to `ic` implicitly
            self.model_result = var.fit(maxlags=maxlags, ic="aic")
            self.model = var
            # prepare data for prediction
            self.lag_order = self.model_result.k_ar
            self.last_observations = train_rtn_df.iloc[-self.lag_order:].values
            self.train_df_last_row = train_df.iloc[[-1], :].copy()

    def predict(self, data: pd.DataFrame, pred_path: Optional[str] = None) \
        -> pd.DataFrame:
        """
        Day by day forecast
        """
        if self.model_type == "VECM":
            pred_list = []
            nd = np.vstack([self.last_observations, data.values])
            for i in range(data.shape[0]):
                pred_i = self.model_result.predict(
                    steps=1, last_observations=nd[i: i + self.lag_order]
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
                    rtn_nd[i: i + self.lag_order], steps=1
                )
                rtn_pred_list.append(rtn_pred_i)
            rtn_pred = np.vstack(rtn_pred_list)
            pred = (1 + rtn_pred).cumprod(axis=0) * \
                self.train_df_last_row.values

        pred_df = pd.DataFrame(pred, index=data.index, columns=data.columns)
        if pred_path:
            pred_df.to_pickle(pred_path + "linear_pred.pkl")
        return pred_df

    def periodic_train_predict(self, data: pd.DataFrame, \
        params: Optional[Mapping] = None) -> pd.DataFrame:
        pass


if __name__ == "__main__":
    # TODO: Load Pairs
    pair_df = pd.read_...

    # Train Test Split
    train_df = pair_df.loc[pair_df.index < "2017-01-01"]
    test_df = pair_df.loc[pair_df.index >= "2017-01-01"]

    # Build Predictor
    predictor = LinearPredictor()
    predictor.train(train_df)
    pred_df = predictor.predict(test_df)

