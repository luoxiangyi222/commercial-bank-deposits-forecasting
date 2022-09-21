from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score
import numpy as np


def RMSE(real_y, predict_y):
    rmse = mean_squared_error(real_y, predict_y, squared=False)
    return rmse


def R2(real_y, predict_y):
    R2 = r2_score(real_y, predict_y)
    return R2


def MPE(real_y, predict_y):
    """
    Mean Percentage Error
    """
    mpe = np.mean((predict_y - real_y) / real_y)
    return mpe


def MAPE(real_y, predict_y):
    """
    Mean Absolute Percentage Error
    """
    mape = mean_absolute_percentage_error(real_y, predict_y)
    return mape


# def get_cv_results(model, df, y_name, x_names, scoring=None, random_state=1):
#     def mean_percentage_error_scorer(cv_mod, X, y):
#         y_pred = cv_mod.predict(X)
#         mpe = ((y_pred - y) / y).mean()
#         return mpe
#
#     if not scoring:
#         scoring = {
#             "r2": "r2",
#             "neg_RMSE": "neg_root_mean_squared_error",
#             "neg_MAPE": "neg_mean_absolute_percentage_error",
#             "MPE": mean_percentage_error_scorer,
#         }
#     # prepare X and y
#     X = df[x_names]
#     y = df[[y_name]]
#
#     # get k-fold
#     kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
#
#     # get cv results
#     cv_results = cross_validate(
#         model,
#         X,
#         y,
#         cv=kf,
#         scoring=scoring,
#         return_train_score=True,
#         return_estimator=True,  # return estimator to compute mean percentage error
#     )
#     ## get consolidated cv result
#     selected_metrics = [
#         "train_R2",
#         "test_R2",
#         "train_RMSE",
#         "test_RMSE",
#         "train_MPE",
#         "test_MPE",
#         "train_MAPE",
#         "test_MAPE",
#     ]
#
#     # R1, RMSE, MAPE
#     agg_results = dict.fromkeys(selected_metrics)
#     agg_results["train_R2"] = cv_results["train_r2"].mean()
#     agg_results["test_R2"] = cv_results["test_r2"].mean()
#     agg_results["train_RMSE"] = -cv_results["train_neg_RMSE"].mean()
#     agg_results["test_RMSE"] = -cv_results["test_neg_RMSE"].mean()
#     agg_results["train_MPE"] = cv_results["train_MPE"].mean()
#     agg_results["test_MPE"] = cv_results["test_MPE"].mean()
#     agg_results["train_MAPE"] = -cv_results["train_neg_MAPE"].mean()
#     agg_results["test_MAPE"] = -cv_results["test_neg_MAPE"].mean()
#
#     # MPE
#
#     return agg_results