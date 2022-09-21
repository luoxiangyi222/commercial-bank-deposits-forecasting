from sklearn.model_selection import KFold
import itertools
import pandas as pd
import ast


def dict_to_dataframe(a_dict):
    res_df = pd.DataFrame(a_dict, index=[0])
    return res_df



def select_k(list_like, k: int, return_index=False):

    if return_index:
        ls_index = range(len(list_like))
        res = itertools.combinations(ls_index, k)
    else:
        res = itertools.combinations(list_like, k)
    return res



def get_k_fold_index_list(df, n_splits=5, random_state=None, shuffle=False):
    """
    Dataset K-fold split
    :param df:
    :param n_splits:
    :param random_state:
    :param shuffle:
    :return:
    """

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    kf.get_n_splits(df)

    k_fold_index_ls = []

    for train_index, test_index in kf.split(df):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_test_pair = {"train_index": train_index, "test_index": test_index}
        k_fold_index_ls.append(train_test_pair)

    return k_fold_index_ls


def listStr_to_list(list_str):
    return ast.literal_eval(list_str)