from math import isnan, sqrt

import numpy as np
import pandas as pd

from scipy import spatial, stats, sparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from surprise import KNNWithMeans
from surprise import Prediction
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

# Currently not used

def similarity_item(dataset_df):
    df_shape = dataset_df.shape
    items = df_shape[1] - 1
    users = df_shape[0] - 1
    is_sparse_matrix = isinstance(dataset_df, sparse.csr_matrix)

    # Create IxI similarity matrices
    item_similarity_cosine = np.zeros((items, items))
    item_similarity_jaccard = np.zeros((items, items))
    item_similarity_pearson = np.zeros((items, items))
    for item1 in range(items):
        for item2 in range(items):
            user_col_item1 = dataset_df.getcol(item1).A.novel() if is_sparse_matrix else dataset_df[:, item1]
            user_col_item2 = dataset_df.getcol(item2).A.novel() if is_sparse_matrix else dataset_df[:, item2]
            if (np.count_nonzero(user_col_item1)
                    and np.count_nonzero(user_col_item2)):
                item_similarity_cosine[item1][item2] = 1 - spatial.distance.cosine(user_col_item1, user_col_item2)
                item_similarity_jaccard[item1][item2] = 1 - (
                    spatial.distance.jaccard(
                        (user_col_item1 > 0.5).astype(int),
                        (user_col_item2 > 0.5).astype(int)
                    )
                )
                try:
                    if not isnan(stats.pearsonr(user_col_item1, user_col_item2)[0]):
                        item_similarity_pearson[item1][item2] = stats.pearsonr(user_col_item1, user_col_item2)[0]
                    else:
                        item_similarity_pearson[item1][item2] = 0
                except:
                    item_similarity_pearson[item1][item2] = 0

    return item_similarity_cosine, item_similarity_jaccard, item_similarity_pearson


def crossValidation(dataset_df):
    k_fold = KFold(n_splits=10, shuffle=True)
    df_shape = dataset_df.shape
    items = df_shape[1]
    users = df_shape[0]

    # U X I matrix probably is not needed
    Mat = np.zeros((users,items))
    for e in dataset_df:
        Mat[e[0]-1][e[1]-1] = e[2]

    sim_item_cosine, sim_item_jaccard, sim_item_pearson = similarity_item(Mat)
    #sim_item_cosine, sim_item_jaccard, sim_item_pearson = np.random.rand(items,items), np.random.rand(items,items), np.random.rand(items,items)

    '''sim_item_cosine = np.zeros((items,items))
    sim_item_jaccard = np.zeros((items,items))
    sim_item_pearson = np.zeros((items,items))
    f_sim_i = open("sim_item_based.txt", "r")
    for row in f_sim_i:
        r = row.strip().split(',')
        sim_item_cosine[int(r[0])][int(r[1])] = float(r[2])
        sim_item_jaccard[int(r[0])][int(r[1])] = float(r[3])
        sim_item_pearson[int(r[0])][int(r[1])] = float(r[4])
    f_sim_i.close()'''

    rmse_cosine = []
    rmse_jaccard = []
    rmse_pearson = []

    for train_indices, test_indices in k_fold.split(dataset_df):
        train = [dataset_df[i] for i in train_indices]
        test = [dataset_df[i] for i in test_indices]

        M = np.zeros((users,items))

        for e in train:
            M[e[0]-1][e[1]-1] = e[2]

        true_rate = []
        pred_rate_cosine = []
        pred_rate_jaccard = []
        pred_rate_pearson = []

        for e in test:
            user = e[0]
            item = e[1]
            true_rate.append(e[2])

            pred_cosine = 3.0
            pred_jaccard = 3.0
            pred_pearson = 3.0

            #item-based
            if np.count_nonzero(M[:,item-1]):
                sim_cosine = sim_item_cosine[item-1]
                sim_jaccard = sim_item_jaccard[item-1]
                sim_pearson = sim_item_pearson[item-1]
                ind = (M[user-1] > 0)
                #ind[item-1] = False
                normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
                normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))
                normal_pearson = np.sum(np.absolute(sim_pearson[ind]))
                if normal_cosine > 0:
                    pred_cosine = np.dot(sim_cosine,M[user-1])/normal_cosine

                if normal_jaccard > 0:
                    pred_jaccard = np.dot(sim_jaccard,M[user-1])/normal_jaccard

                if normal_pearson > 0:
                    pred_pearson = np.dot(sim_pearson,M[user-1])/normal_pearson

            if pred_cosine < 0:
                pred_cosine = 0

            if pred_cosine > 5:
                pred_cosine = 5

            if pred_jaccard < 0:
                pred_jaccard = 0

            if pred_jaccard > 5:
                pred_jaccard = 5

            if pred_pearson < 0:
                pred_pearson = 0

            if pred_pearson > 5:
                pred_pearson = 5

            # print(f"{user}  {item} {e[2]} {pred_cosine} {pred_jaccard} {pred_pearson}")
            pred_rate_cosine.append(pred_cosine)
            pred_rate_jaccard.append(pred_jaccard)
            pred_rate_pearson.append(pred_pearson)

        rmse_cosine.append(sqrt(mean_squared_error(true_rate, pred_rate_cosine)))
        rmse_jaccard.append(sqrt(mean_squared_error(true_rate, pred_rate_jaccard)))
        rmse_pearson.append(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))

        print(f"{sqrt(mean_squared_error(true_rate, pred_rate_cosine))}"
              f"{sqrt(mean_squared_error(true_rate, pred_rate_jaccard))}"
              f"{sqrt(mean_squared_error(true_rate, pred_rate_pearson))}")

    rmse_cosine = sum(rmse_cosine) / float(len(rmse_cosine))
    rmse_pearson = sum(rmse_pearson) / float(len(rmse_pearson))
    rmse_jaccard = sum(rmse_jaccard) / float(len(rmse_jaccard))

    print(f"{rmse_cosine} {rmse_jaccard} {rmse_pearson}")

    f_rmse = open("rmse_item.txt", "w")
    f_rmse.write(f" rmse_cosine: {rmse_cosine}\n rmse_jaccard:{rmse_jaccard}\n rmse_pearson: {rmse_pearson}")

    rmse = [rmse_cosine, rmse_jaccard, rmse_pearson]
    req_sim = rmse.index(min(rmse))

    f_rmse.write(str(req_sim))
    f_rmse.close()

    sim_mat_item = None
    if req_sim == 0:
        sim_mat_item = sim_item_cosine

    if req_sim == 1:
        sim_mat_item = sim_item_jaccard

    if req_sim == 2:
        sim_mat_item = sim_item_pearson

    return Mat, sim_mat_item


def construct_cf_matrix(predictions: [Prediction]):
    baseline_cf_values = []

    for pred in predictions:
        baseline_cf_values.append({
            "movieId": pred.iid,
            "userId": pred.uid,
            "rating": pred.est
        })
    baseline_cf_df = pd.DataFrame(baseline_cf_values)
    # pivot ratings into movie features
    baseline_cf_df_pivot = baseline_cf_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    return baseline_cf_df_pivot


def calculate_baseline_cf(df: pd.DataFrame, verbose=True, **kwargs):
    """

    Args:
        df:
        verbose:
        **kwargs:

    Returns:

    """
    # Load the movielens-100k dataset  UserID::MovieID::Rating::Timestamp
    # data = Dataset.load_builtin('ml-100k')
    # A reader is still needed but only the rating_scale param is requiered.

    rating_scale = kwargs.get("rating_scale", (1, 5))
    important_cols = kwargs.get("important_cols", ["movieId", "userId", "rating"])
    test_size = kwargs.get("test_size", 0.15)
    surprise_kwargs = kwargs.get("surprise_kwargs", {
        "k": 50,
        "sim_options": {
            "name": "pearson_baseline",
            "user_based": False
        }
    })

    assert (set(important_cols) < set(df.colums)), "Important cols are not a subset of the Dataframe"

    reader = Reader(rating_scale=rating_scale)

    data = Dataset.load_from_df(df[important_cols], reader)
    trainset, testset = train_test_split(data, test_size=test_size)

    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    algo = KNNWithMeans(**surprise_kwargs)
    algo.fit(trainset)

    # run the trained model against the testset
    test_pred = algo.test(testset)

    # get RMSE
    if verbose:
        print("Item-based Model : Test Set")
    accuracy.rmse(test_pred, verbose=True)

    # if you wanted to evaluate on the trainset
    if verbose:
        print("Item-based Model : Test Set")
        print("Item-based Model : Training Set")
    train_pred = algo.test(trainset.build_testset())
    accuracy.rmse(train_pred)

    return test_pred + train_pred
