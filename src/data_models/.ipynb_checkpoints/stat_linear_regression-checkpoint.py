import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


def get_SLR_analysis_results(
        X, y, selected_criteria=None,
):
    if selected_criteria is None:
        selected_criteria = ['R-squared',"Adj. R-squared", "AIC", "BIC", "Prob (F-statistic)", ]

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

        self.train_X = train_X
        self.train_y = train_y
        self.train_predict_y = None
        self.N, self.D = train_X.shape

        # print(self.N, self.D)
        # train linear regression model
        self.train()

    def train(self):
        # !!! Don't forget to add bias column
        self.model = sm.OLS(self.train_y, add_constant(self.train_X))
        self.fitted_model = self.model.fit()
        self.train_predict_y = self.fitted_model.predict(add_constant(self.train_X))
        self.train_result_tables = self.fitted_model.summary2().tables
        

    def predict(self, X):
        X = sm.add_constant(X)
        return self.fitted_model.predict(X)

    def get_train_summary(self):
        summary = self.fitted_model.summary2()
        return summary

    def get_train_eval(self, selected_metrics: list = None):
        res_df = self.train_result_tables[0]

        part1 = res_df[[0, 1]]
        part2 = res_df[[2, 3]]
        part2.columns = part1.columns
        res_df = pd.concat([part1, part2])

        res_df = res_df.T
        res_df.columns = res_df.iloc[0].apply(lambda s: s.strip().replace(":", ""))
        res_df = res_df.iloc[1:]

        compulsory_cols = list(res_df.columns[:6])
        
        # display(res_df)

        if selected_metrics:
            res_df = res_df[compulsory_cols + selected_metrics]

        return res_df

    def get_res_tables(self):
        return self.train_result_tables

    def adj_R2(self, real_y, predict_y):
        R2 = self.R2(real_y, predict_y)
        adj_r2_score = 1 - (1 - R2) * (self.N - 1) / (self.N - self.D - 1)
        return adj_r2_score


