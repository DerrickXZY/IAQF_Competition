import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def performance_for_series(rtn_s, rf=0.0):
    """
    Output:
        NAV_s: pd.Series, Net Asset Value (NAV) Series
        Simple Return, Annualized Return, Annualized Volatility, Sharpe Ratio, 
        Adjusted Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio
    """
    Rtn_s = rtn_s + 1
    NAV_s = Rtn_s.cumprod()
    trading_len = len(rtn_s)

    # Simple Return
    rtn = NAV_s.iloc[-1] - 1

    # Annualized Return
    annualized_rtn = (1 + rtn) ** (252 / trading_len) - 1
    
    # Annualized Volatility
    annualized_vol = rtn_s.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (annualized_rtn - rf) / annualized_vol

    # Adjusted Sharpe Ratio
    skewness = rtn_s.skew()
    kurtosis = rtn_s.kurt()
    adj_sharpe = sharpe * (1 + (skewness / 6) * sharpe 
        - (kurtosis - 3) / 24 * (sharpe ** 2))

    # Sortino Ratio
    down_rtn_s = rtn_s[rtn_s < rf]
    down_rtn_annualized_vol = down_rtn_s.std() * np.sqrt(252)
    sortino = (annualized_rtn - rf) / down_rtn_annualized_vol

    # Maximum Drawdown
    # If maximum drawdown series is needed, just uncomment the code below
    max_drawdown = 0.0
    # max_drawdown_l = [0.0]
    max_price = NAV_s.iloc[0]
    for price in NAV_s.iloc[1:]:
        if price > max_price:
            max_price = price
        else:
            drawdown = price / max_price - 1
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        # max_drawdown_l.append(max_drawdown)
    max_drawdown = abs(max_drawdown)
    # max_drawdown_s = pd.Series(max_drawdown_l, index=rtn_s.index)

    # Calmar Ratio
    calmar = annualized_rtn / max_drawdown

    result_s = pd.Series(
        [rtn, annualized_rtn, annualized_vol, sharpe, adj_sharpe, sortino, 
        max_drawdown, calmar],                
        index=["Simple Return", "Annualized Return", "Annualized Volatility", 
        "Sharpe Ratio", "Adjusted Sharpe Ratio", "Sortino Ratio", 
        "Maximum Drawdown", "Calmar Ratio"],
        name=rtn_s.name
    )
    return NAV_s, result_s


def performance_for_df(rtn_df, rf=0.0):
    NAV_l = []
    result_l = []
    for j in range(rtn_df.shape[1]):
        NAV_s, result_s = performance_for_series(rtn_df.iloc[:, j], rf)
        NAV_l.append(NAV_s)
        result_l.append(result_s)
    NAV_df = pd.concat(NAV_l, axis=1)
    result_df = pd.concat(result_l, axis=1)
    return NAV_df, result_df


def judge_overlap(x_y, Range_y, len_x, T_y, T_x):
    """
    Input:
        x_y: pd.Series, index is x, value is y, x and y are both numerics and 
            stand for x-coordinate and y-coordinate, x_y has been sorted by y
    Output:
        If there exists an overlap, return the index of the first item needed 
        to fix; otherwise, return None
    """
    y_diff_ratio = np.diff(x_y.values) / Range_y
    overlap_y = (np.abs(y_diff_ratio) + 1e-8) < T_y
    if overlap_y.any():
        x_diff_ratio = np.diff(x_y.index.to_numpy()) / len_x
        overlap_x = x_diff_ratio < T_x
        overlap = overlap_y & overlap_x
        if overlap.any():
            idx = overlap.tolist().index(True)
            return idx + 1 # as np.diff drops the first np.nan, so add 1
    return None


def adjust_overlap(x_y, adjust_idx, adj_unit):
    """
    Input:
        x_y: pd.Series, index is x, value is y, x and y are both numerics and 
            stand for x-coordinate and y-coordinate, x_y has been sorted by y
        adjust_idx: int(>=1), index of x_y needed to adjust
        adj_unit: float, correction magnitude of y
    Output:
        Adjusted x_y
    """
    x_y_copy = x_y.copy()
    x_y_copy.iloc[adjust_idx] = x_y_copy.iloc[adjust_idx-1] + adj_unit
    return x_y_copy


def NAV_df_plot(NAV_df, high_mark_cols, low_mark_cols, labels=[],
                grid=True, fig_path=None):
    """
    Input:
        high_mark_cols: list, indices/funds you want to add high annotation
        low_mark_cols: list, indices/funds you want to add low annotation
        labels: list, lables in legend
    """
    plt.plot(NAV_df)
    plt.xticks(rotation=30)
    plt.xlabel("Time")
    plt.ylabel("NAV")
    range_nav = NAV_df.values.max() - NAV_df.values.min() # range of NAV
    len_t = NAV_df.shape[0]
    for mark_cols,fun,argfun,label,color in zip([high_mark_cols,low_mark_cols],
                                                [np.max,np.min],
                                                [np.argmax,np.argmin],
                                                ["High","Low"],
                                                ["green","red"]):
        if mark_cols:
            y_s = NAV_df[mark_cols].apply(fun)
            x_s = NAV_df[mark_cols].apply(argfun)
            # xynav is the coordinate of high/low
            xynav = pd.Series(y_s.values, index=x_s.values)
            # prevent a situation that reversed xytext cannot match xynav 
            # according to index when high and low of different series occur
            # at the same time
            real_index = xynav.index
            xynav.reset_index(drop=True, inplace=True)
            # xytext is the coordinate of text
            xytext = xynav.copy()
            # judge whether it would be too close for two text coordiates.
            # if so, fix it
            xytext.sort_values(ascending=False, inplace=True)
            adj_idx = judge_overlap(xytext, range_nav, len_t, T_y=0.04, T_x=0.15)
            while (adj_idx is not None):
                xytext = adjust_overlap(xytext, adj_idx, -0.07*range_nav)
                adj_idx = judge_overlap(xytext, range_nav, len_t, T_y=0.04, T_x=0.15)
            # annotate high/low
            xy_nav_text = pd.concat([xynav, xytext], axis=1)
            xy_nav_text.index = real_index
            xy_nav_text.columns = ["y_nav", "y_text"]
            for i, idx in enumerate(xy_nav_text.index):
                t = NAV_df.index[idx]
                y_nav = xy_nav_text.y_nav.iloc[i]
                y_text = xy_nav_text.y_text.iloc[i]
                xy = (t, y_nav)
                xytext = (t, y_text)
                plt.annotate(label+":%.3f" % y_nav, xy=xy, xytext=xytext,
                            arrowprops={
                                "facecolor":color,
                                # arrow breadth (unit: dot)
                                "width": 0, 
                                # arrowhead breadth (unit: dot)
                                "headwidth": 5, 
                                # arrowhead length (unit: dot)
                                "headlength": 5, 
                                # arrowhead side shrinkage percentage 
                                # (of total length)
                                # "shrink": 0.5, 
                            })
    if labels:
        plt.legend(labels=labels)
    plt.grid(grid)
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", dpi=480)
    # plt.show()
    

def NAV_s_plot(NAV_s, high_mark=True, low_mark=True, grid=True, fig_path=None):
    NAV_s.plot.line()
    plt.xlabel("Time")
    plt.ylabel("NAV")
    # annotate high/low
    if high_mark:
        maxNAV = NAV_s.max()
        t_maxNAV = NAV_s.index[NAV_s.argmax()]
        # plt.text(t_maxNAV, maxNAV, "High: %.3f" % maxNAV)
        max_xy = (t_maxNAV, maxNAV)
        max_xytext = max_xy
        plt.annotate("High: %.3f" % maxNAV, xy=max_xy, xytext=max_xytext,
                      arrowprops={"facecolor":"green","shrink":0,"width":0})
    if low_mark:
        minNAV = NAV_s.min()
        t_minNAV = NAV_s.index[NAV_s.argmin()]
        # plt.text(t_minNAV, minNAV, "Low: %.3f" % minNAV)
        min_xy = (t_minNAV, minNAV)
        min_xytext = min_xy
        plt.annotate("Low: %.3f" % minNAV, xy=min_xy, xytext=min_xytext,
                      arrowprops={"facecolor":"red","shrink":0,"width":0})
    plt.grid(grid)
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", dpi=480)
        


if __name__ == "__main__":
    import yfinance as yf
    import os
    os.chdir(os.path.dirname(__file__))

    start_date = pd.Timestamp("2000-01-01")
    end_date = pd.Timestamp("2019-12-31")

    spy = yf.download("SPY", start=start_date, end=end_date)
    r2k = yf.download("IWM", start=start_date, end=end_date)
    rtn_df = pd.concat([spy["Adj Close"].pct_change().dropna(), 
                        r2k["Adj Close"].pct_change().dropna()],
                        axis=1, join="inner")
    rtn_df.columns = ["SPY", "R2K"]
    nav_df, result_df = performance_for_df(rtn_df, rf=0.0)
    print(result_df)
    NAV_df_plot(nav_df, nav_df.columns.tolist(), nav_df.columns.tolist(),
                labels=["spy_demo", "r2k_demo"], fig_path="demo_NAV_figure.jpg")
