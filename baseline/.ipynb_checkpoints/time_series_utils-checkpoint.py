# Note: Tools in this file only supports np.ndarray but not pd.Series nor pd.DataFrame
import numpy as np
from scipy.stats import norm, chi2

from scipy.linalg import toeplitz
# from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.adfvalues import mackinnonp


class OLS:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        self.y_hat = X @ self.params
        self.e = y - self.y_hat
        self.n, self.k = X.shape
        self.k -= 1
        self.R2 = None
        self.V_hat = None
        self.params_std = None
        self.V_white_hat = None
        self.params_white_std = None
        self.logL = None
    
    def get_R2(self):
        SSR = (self.e ** 2).sum()
        SST = ((self.y - self.y.mean()) ** 2).sum()
        self.R2 = 1 - SSR / SST
        return self.R2
    
    def get_adjusted_R2(self):
        if self.R2 is None:
            self.get_R2()
        adjusted_R2 = 1 - (1 - self.R2) * (self.n - 1) / (self.n - self.k - 1)
        return adjusted_R2
    
    def get_V_hat(self):
        sigma2_hat = 1 / (self.n - self.k) * (self.e**2).sum()
        self.V_hat = np.linalg.inv(self.X.T @ self.X) * sigma2_hat
        return self.V_hat
    
    def get_params_std(self):
        if self.V_hat is None:
            self.get_V_hat()
        self.params_std = np.sqrt(self.V_hat.diagonal())
        return self.params_std
    
    def t_test(self):
        if self.params_std is None:
            self.get_params_std()
        t_values = self.params / self.params_std
        p_values = np.where(t_values > 0,
                            (1 - norm.cdf(t_values)) * 2, 
                            norm.cdf(t_values) * 2
                           )
        return t_values, p_values
    
    def white_test(self):
        u2 = self.e ** 2
        X_noconst = self.X[:,1:]
        X_white = self.X.copy()
        for j in range(X_noconst.shape[1]):
            X_white = np.hstack([X_white, X_noconst[:,[j]] * X_noconst[:,j:]])
        ols_white = OLS(X_white, u2)
        R2_white = ols_white.get_R2()
        TR2 = ols_white.n * R2_white
        p_value_white = 1 - chi2.cdf(TR2, df=ols_white.k)
        return TR2, p_value_white
    
    def get_V_white_hat(self):
        D = np.diag(self.e ** 2)
        self.V_white_hat = np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ D @ self.X) @ np.linalg.inv(self.X.T @ self.X)
        return self.V_white_hat # white variance-covariance matrix
    
    def get_params_white_std(self):
        if self.V_white_hat is None:
            self.get_V_white_hat()
        self.params_white_std = np.sqrt(np.diagonal(self.V_white_hat))
        return self.params_white_std
    
    def t_test_white(self):
        if self.params_white_std is None:
            self.get_params_white_std()
        t_values_white = self.params / self.params_white_std
        p_values_white = np.where(t_values_white > 0,
                                  2 * (1 - norm.cdf(t_values_white)),
                                  2 * norm.cdf(t_values_white)
                                 )
        return t_values_white, p_values_white
    
    def get_logL(self):
        sigma2_hat = 1 / (self.n - self.k - 1) * (self.e**2).sum()
        self.logL = - self.n / 2 * np.log(2 * np.pi) - \
                    self.n / 2 * np.log(sigma2_hat) - \
                    (self.e ** 2).sum() / (2 * sigma2_hat)
        return self.logL
    
    def get_AIC(self):
        if self.logL is None:
            self.get_logL()
        AIC = - 2 * self.logL + 2 * self.k
        return AIC
        
    def get_BIC(self):
        if self.logL is None:
            self.get_logL()
        BIC = - 2 * self.logL + self.k * np.log(self.n)
        return BIC
        
    def get_HQ(self):
        if self.logL is None:
            self.get_logL()
        HQ = - 2 * self.logL + 2 * self.k * np.log(np.log(self.n))
        return HQ
    
    def DW_test(self):
        ut = self.e[1:]
        ut_1 = self.e[:-1]
        DW = ((ut - ut_1) ** 2).sum() / (ut ** 2).sum()
        return DW
    
    def BG_test(self, lag=3):
        X_BG = np.full((len(self.e),lag), np.nan)
        for j in range(1, lag + 1):
            X_BG[j:,j-1] = self.e[:-j]
        u = self.e[lag:]
        X_BG = X_BG[lag:,:]
        ols_BG = OLS(X_BG, u)
        R2_BG = ols_BG.get_R2()
        T_r_R2 = (self.n - lag) * R2_BG # ~ chi2(r)
        p_value_BG = 1 - chi2.cdf(T_r_R2, df=lag)
        return T_r_R2, p_value_BG
    
    def qqplot_e(self):
        cdfs = (np.arange(len(self.e)) + 0.5) / len(self.e) # 0 ~ 1 ex: if there ara 100 hunred points, it will be like 0.01, 0.02, ...
        theoretical_distribution = norm.ppf(cdfs) # from the probability to the value in the normal distribution ex: 0.975->1.96
        fig = plt.figure()
        plt.scatter(theoretical_distribution, np.sort(self.e), s=15)
        benchmark_line = [min(self.e.min(), theoretical_distribution.min()),
                          max(self.e.max(), theoretical_distribution.max())]
        plt.plot(benchmark_line, benchmark_line, c="r")
        plt.xlabel("Quantile of normal distribution")
        plt.ylabel("Quantile of residuals")
        plt.show()
    
    def get_cond(self):
        return np.linalg.cond(self.X.T @ self.X)
    
    def RESET_test(self):
        X1 = np.hstack([self.X, (self.y_hat ** 2).reshape((-1,1))])
        ols_RESET = OLS(X1, y)
        t_values_RESET, p_values_RESET = ols_RESET.t_test()
        t_value_yhat2 = t_values_RESET[-1]
        p_value_yhat2 = p_values_RESET[-1]
        return t_value_yhat2, p_value_yhat2
    
    def stability_analysis(self, window, t=None):
        betas = []
        upper_bounds = []
        lower_bounds = []
        for i in range(self.X.shape[0] - window):
            X_i = self.X[i:i+window,:]
            y_i = self.y[i:i+window]
            ols_stab = OLS(X_i, y_i)
            beta_i = ols_stab.params[1]
            betas.append(beta_i)
            beta_i_std = ols_stab.get_params_std()[1]
            lower_bounds.append(beta_i + norm.ppf(0.025) * beta_i_std)
            upper_bounds.append(beta_i + norm.ppf(0.975) * beta_i_std)
        
        if t is None:
            t = np.arange(len(betas))
        fig = plt.figure()
        plt.plot(t, betas)
        plt.fill_between(t, lower_bounds, upper_bounds, alpha=0.3)
        plt.xlabel("t")
        plt.ylabel("beta estimate")
        plt.show()
        
    def predict(self, X_new):
        return X_new @ self.params
    

class TSA:
    def __init__(self, x):
        self.x = x
    
    # ACF & PACF
    def _lag_mat(self, nlags, x=None):
        if x is None:
            x = self.x
        x_lags = np.full((len(x), nlags + 1), np.nan)
        # include 0 lag
        for lag in range(nlags + 1):
            x_lags[lag:, lag] = np.roll(x, shift=lag)[lag:]
        return x_lags
    
    def cal_acf_lag(self, lag):
        if lag == 0:
            return 1.
        return np.corrcoef(self.x[lag:], self.x[:-lag])[1,0]
    
    def cal_acf(self, nlags):
        lag_mat = self._lag_mat(nlags)
        acf_l = [1] + [np.corrcoef(lag_mat[lag:,0], lag_mat[lag:,lag])[1,0] for lag in range(1, nlags + 1)]
        return np.array(acf_l)
    
    def cal_pacf_lag(self, lag):
        # should be the same with sm.tsa.stattools.acf
        # Method1:
        # Reference: https://zhuanlan.zhihu.com/p/430514606
        x = self.x - self.x.mean()
        x_prime = x.copy()
        x = x.reshape((-1,1))
        n = len(x)
        
        gamma_l = []
        gamma_l.append((x_prime @ x) / n) # gamma(0)
        for i in range(1, lag + 1):
            gamma_l.append((x_prime[i:] @ x[:-i]) / (n - i)) # gamma(lag)
        
        R_mat = toeplitz(gamma_l[:-1])
        # solve R_mat @ phi_hat = gamma_arr
        phi_hat = np.linalg.inv(R_mat) @ np.array(gamma_l[1:]).reshape((-1,1))
        return phi_hat[-1].item()
        # Method2:
        # run the following ols regression
        # x_t = beta_0 + beta_1*x_t-1 + beta_2*x_t-2 + ... + beta_lag*x_t-lag + e_t
        # beta_lag is pacf for lag n
    
    def cal_pacf(self, nlags):
        # should be the same with sm.tsa.stattools.pacf
        pacf_l = [1]
        for lag in range(1, nlags + 1):
            pacf_l.append(self.cal_pacf_lag(lag))
        return np.array(pacf_l)
        
    def plot_acf(self, nlags, acf_arr=None, ax=None):
        # should be the same with statsmodels.graphics.tsaplots.plot_acf
        if acf_arr is None:
            acf_arr = self.cal_acf(nlags)
        # For details: cf. slide 23 of Lecture 1 and part on bartlett_confin on
        # https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
        n = len(self.x)
        se_approx = np.sqrt((np.insert(1 + 2 * np.cumsum(acf_arr ** 2), 0, 1, axis=0)[:-1]) / n)
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        ax.plot(acf_arr, c='xkcd:true blue', marker='o', markerfacecolor='xkcd:azure')
        ax.fill_between(x=range(nlags+1), y1=-1.96*se_approx, y2=1.96*se_approx, facecolor='blue', alpha=0.1)
        ax.set_xlabel("lag")
        ax.axhline(y=0, linewidth=0.4)
        ax.set_title("ACF Figure")
        
    def plot_pacf(self, nlags, pacf_arr=None, ax=None):
        # should be the same with statsmodels.graphics.tsaplots.plot_pacf
        if pacf_arr is None:
            pacf_arr = self.cal_pacf(nlags)
        n = len(self.x)
        se = np.ones(nlags + 1) * np.sqrt(1 / n)
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        ax.plot(pacf_arr, c='xkcd:true blue', marker='o', markerfacecolor='xkcd:azure')
        ax.fill_between(x=range(nlags+1), y1=-1.96*se, y2=1.96*se, facecolor='blue', alpha=0.1)
        ax.set_xlabel("lag")
        ax.axhline(y=0, linewidth=0.4)
        ax.set_title("PACF Figure")

    # Stationarity test: ADF test
    def _get_ADF_X_y(self, lag, terms="n"):
        delta_x = self.x[1:] - self.x[:-1]
        delta_lag_mat = self._lag_mat(lag, delta_x)
        y = delta_lag_mat[lag:, 0] # trancate
        X_delta_lag = delta_lag_mat[lag:, 1:] # trancate
        xt_1 = self.x[:-1]
        Xt_1 = xt_1[lag:].reshape((-1,1)) # trancate
        if terms == "ctt":
            n_ = len(y)
            constant_term = np.ones((n_,1))
            trend_term = np.arange(lag + 1, n_ + lag + 1).reshape((-1,1))
            trend_square_term = trend_term ** 2
            X = np.hstack([Xt_1, X_delta_lag, constant_term, trend_term, trend_square_term])
        elif terms == "ct":
            n_ = len(y)
            constant_term = np.ones((n_,1))
            trend_term = np.arange(lag + 1, n_ + lag + 1).reshape((-1,1))
            X = np.hstack([Xt_1, X_delta_lag, constant_term, trend_term])
        elif terms == "c":
            n_ = len(y)
            constant_term = np.ones((n_,1))
            X = np.hstack([Xt_1, X_delta_lag, constant_term])
        elif terms == "n":
            X = np.hstack([Xt_1, X_delta_lag])
        return X, y

    def _are_terms_significant(self, n_terms, ols, maxlag, criterion):
        assert n_terms >= 1, "n_terms should be at no less than 1"
        mapping_dict = {3: "ctt", 2: "ct", 1: "c", 0: "n"}
        terms = mapping_dict[n_terms]
        
        terms_ADF_stats = ols.params[-n_terms:] / ols.get_params_std()[-n_terms:]
        ADF_p_values = np.array([mackinnonp(adf_stat, terms, N=1) for adf_stat in terms_ADF_stats])
        if (ADF_p_values < 0.05).all():
            # print("Reject null hypothesis. No evidence shows that it has a unit root.")
            # print("The time series is STATIONARY.")
            return True
        else:
            # print("Cannot reject the null hypothesis that it has a unit root")
            # print("The time series is UNSTATIONARY.")
            return False
        
        # OLD VERSION CODE (CAN BE REMOVED)
        # # get ADF critical values
        # result = adfuller(self.x, maxlag, terms, criterion)
        # critical_values = result[4]
        # if (critical_values["1%"] > critical_values["5%"] and (terms_ADF_stats > critical_values["5%"]).all()) \
        # or (critical_values["1%"] < critical_values["5%"] and (terms_ADF_stats < critical_values["5%"]).all()):
        #     # print("Reject null hypothesis. No evidence shows that it has a unit root.")
        #     # print("The time series is STATIONARY.")
        #     return True
        # else:
        #     # print("Cannot reject the null hypothesis that it has a unit root")
        #     # print("The time series is UNSTATIONARY.")
        #     return False
    
    def ADF_regression(self, maxlag=5, terms="n", criterion="AIC"):
        """
        maxlag: max lag attempt
        regression: "n" - no constant nor trend term
                    "c" - only constant term
                    "ct" - constant and trend (i.e. t) term
                    "ctt" - constant, trend and quadratic trend term
        criterion: "AIC", "BIC" - the criterion to choose the best model (best lag)
                   to judge whether the time series is stationary or not
        """
        cri_l = [] # record criterion value for each lag
        for lag in range(1, maxlag + 1):
            X, y = self._get_ADF_X_y(lag, terms)
            ols = OLS(X, y)
            if criterion == "AIC":
                cri = ols.get_AIC()
            elif criterion == "BIC":
                cri = ols.get_BIC()
            cri_l.append(cri)
        best_lag = np.argmin(cri_l) + 1
        # print(f"The best lag for ADF regression is {best_lag}")
        X, y = self._get_ADF_X_y(best_lag, terms)
        ols = OLS(X, y)
        return ols, best_lag
    
    def ADF_test(self, maxlag=5, terms="n", criterion="AIC", ols=None):
        if ols is None:
            ols, best_lag = self.ADF_regression(maxlag, terms, criterion)
        else:
            best_lag = None
        rho_hat = ols.params[0]
        rho_std = ols.get_params_std()[0]
        ADF_stats = rho_hat / rho_std # or t_values, _ = ols.t_test(); ADF_stats = t_values[0]
        ADF_p_value = mackinnonp(ADF_stats, terms, N=1)        
        
        # OLD VERSION CODE (CAN BE REMOVED)
        # result = adfuller(self.x, maxlag, terms, criterion)
        # # adf_test_statistics = result[0] # adf statistics: rho_hat / se(rho_hat)
        # ADF_p_value = result[1] # adf statistics p-value
        # # usedlag = result[2] # the same with `best_lag`
        # # nob = result[3] # number of objects
        # # critical_values = result[4] # dict: critical values for 1%, 5%, and 10%
        
        return ADF_stats, ADF_p_value, best_lag

    def ADF_test_complete(self, maxlag=5, criterion='AIC'):
        """
        maxlag: max lag attempt
        criterion: "AIC", "BIC" - the criterion to choose the best model (best lag)
                   to judge whether the time series is stationary or not
        """
        # first determine the terms to add: "ctt" or "ct" or "c" or "n"
        # then conduct ADF test with proper terms
        mapping_dict = {3: "ctt", 2: "ct", 1: "c", 0: "n"}
        n_terms = 3
        terms = mapping_dict[n_terms]
        ols, best_lag = self.ADF_regression(maxlag, terms, criterion)
        while not self._are_terms_significant(n_terms, ols, maxlag, criterion):
            n_terms -= 1
            terms = mapping_dict[n_terms]
            ols, best_lag = self.ADF_regression(maxlag, terms, criterion)
            if n_terms == 0:
                break
        ADF_stats, ADF_p_value, _ = self.ADF_test(maxlag, terms, criterion, ols)
        return ADF_stats, ADF_p_value, best_lag, terms
    
    def AR(self, p):
        lag_mat = self._lag_mat(p)
        y = lag_mat[p:, 0]
        X = lag_mat[p:, 1:]
        return OLS(X, y)