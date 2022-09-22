import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from src.data_models.model_eval import *


def get_SLR_analysis_results(
        X, y, selected_criteria=None,
):
    if selected_criteria is None:
        selected_criteria = ['R-squared', "Adj. R-squared", "AIC", "BIC", "Prob (F-statistic)", ]

    df_to_concat = []
    original_y = y.copy()

    for i, col_name in enumerate(X.columns):
        # print("==================" + col_name + "==================")
        single_X = X[[col_name]].loc[X[col_name].notnull()]
        single_X = np.array(single_X)

        current_y = original_y.loc[X[col_name].notnull()]

        # print(single_X.shape)
        # print(current_y.shape)

        mySLR = MyLR(train_X=single_X, train_y=current_y)

        eval_df = mySLR.get_train_eval(selected_criteria)
        # display(eval_t)
        # model_t = mySLR.get_model()

        # create alias for independent variables
        eval_df["x_name"] = col_name
        eval_df["x_alias"] = f"V_{i + 1}"

        df_to_concat.append(eval_df)

        # display(current_res_df)

    res_df = pd.concat(df_to_concat).reset_index(drop=True)

    res_df[selected_criteria] = res_df[selected_criteria].astype("float64")

    # print("!!!!" + str(res_df.shape))

    return res_df


class MyLR:
    def __init__(self, train_X, train_y):
        self.model = None
        self.fitted_model = None
        self.train_result_tables = None

        # train
        self.train_X = train_X
        self.train_N, self.D = train_X.shape
        self.train_y = train_y
        self.train_pred_y = None

        # test
        self.test_X = None
        self.test_y = None
        self.test_pred_y = None

        # evaluation
        self.eval_df = None

        # train linear regression model
        self.train()

    def train(self):
        # !!! Don't forget to add bias column
        self.model = sm.OLS(self.train_y, add_constant(self.train_X))
        self.fitted_model = self.model.fit()
        self.train_pred_y = self.fitted_model.predict(add_constant(self.train_X))
        self.train_result_tables = self.fitted_model.summary2().tables

    def predict(self, X):
        X = sm.add_constant(X)
        return self.fitted_model.predict(X)

    def forecast(self, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y
        self.test_pred_y = self.predict(test_X)
        return self.test_pred_y

    def get_train_summary(self):
        summary = self.fitted_model.summary2()
        return summary

    def add_test_eval(self):
        """
        Forecast function should have been called to ensure the test_pred_y exist, otherwise should return False.
        :return: [bool] to show whether evaluation metrics for test dataset has been added.
        """
        if self.test_pred_y:
            self.eval_df["test_RMSE"] = RMSE(self.test_y, self.test_pred_y)
            self.eval_df["test_R2"] = R2(self.test_y, self.test_pred_y)
            self.eval_df["test_MPE_1y"] = MPE(self.test_y[:4], self.test_pred_y[:4])
            self.eval_df["test_MPE_2y"] = MPE(self.test_y[:8], self.test_pred_y[:8])
            self.eval_df["test_MPE_3y"] = MPE(self.test_y[:12], self.test_pred_y[:12])
            self.eval_df["test_MAPE_1y"] = MAPE(self.test_y[:4], self.test_pred_y[:4])
            self.eval_df["test_MAPE_2y"] = MAPE(self.test_y[:8], self.test_pred_y[:8])
            self.eval_df["test_MAPE_3y"] = MAPE(self.test_y[:12], self.test_pred_y[:12])
            return True
        else:
            return False

    def add_train_eval(self, selected_metrics: list = None):
        """
        This should be called after self.train().Compute evaluation metrics for training dataset.
        """
        if self.train_result_tables:
            res_df = self.train_result_tables[0]

            # format the dataframe into two columns/rows
            part1 = res_df[[0, 1]]
            part2 = res_df[[2, 3]]
            part2.columns = part1.columns
            res_df = pd.concat([part1, part2])

            res_df = res_df.T
            res_df.columns = res_df.iloc[0].apply(lambda s: s.strip().replace(":", ""))
            res_df = res_df.iloc[1:]

            compulsory_cols = list(res_df.columns[:6])

            # add customized metrics into eval df
            res_df['train_RMSE'] = np.sqrt(mean_squared_error(self.train_y, self.train_pred_y))
            res_df['train_R2'] = r2_score(self.train_y, self.train_pred_y)
            res_df['train_MPE'] = MPE(self.train_y, self.train_pred_y)
            res_df['train_MAPE'] = mean_absolute_percentage_error(self.train_y, self.train_pred_y)

            if selected_metrics:
                res_df = res_df[compulsory_cols + selected_metrics]

            # save it as class variable
            self.eval_df = res_df
            return True
        else:
            return False



