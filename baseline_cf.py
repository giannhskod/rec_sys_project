import os
import pandas as pd

from surprise import KNNWithMeans, Prediction, accuracy, Dataset, Reader
from surprise.model_selection import train_test_split

from definitions import MODELS_DIR

BASELINE_CF_PICKLE_FILE_NAME = os.path.join(MODELS_DIR, "baseline-cf-pickle")


def construct_cf_matrix(predictions: [Prediction]):
    """
    Constructs the Collaborative Filtering matrix based on the given "surprise.Prediction" list.
    The matrix is generated from the "estimated" values
    Args:
        predictions (list): A list of surprise.Prediction items.

    Returns (pandas.DataFrame): Is the CF DataFrame with 'userId' as index 'movieId' as columns and estimated
                                ratings as values.
    """
    baseline_cf_values = []

    for pred in predictions:
        baseline_cf_values.append(
            {"movieId": pred.iid, "userId": pred.uid, "rating": pred.est}
        )
    baseline_cf_df = pd.DataFrame(baseline_cf_values)

    # pivot ratings into movie features
    baseline_cf_df_pivot = baseline_cf_df.pivot(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)
    return baseline_cf_df_pivot


def calculate_baseline_cf(df: pd.DataFrame, verbose=True, **kwargs):
    """
    Function that given a User - Item - Rating DataFrame and a set of dynamic arguments,
    it create a prediction Algorithm model using KNN neighborhood method.
    For the Model construction and the prediction, we use the 'surprise' scikit and more
    specifically the KNNwithMeans object, which is initialized based on the given arguments.

    Args:
        df (pandas.Dataframe): The User - Item - Rating DataFrame
        verbose (bool): If true the print execution info
        **kwargs (dict): Dynamic arguments that will be used from the Prediction Model initialization
                         and the data management.

    Returns (surprise.Prediction): A list of the predicted values.

    """
    rating_scale = kwargs.get("rating_scale", (1, 5))
    important_cols = kwargs.get("important_cols", ["userId", "movieId", "rating"])
    test_size = kwargs.get("test_size", 0.15)
    surprise_kwargs = kwargs.get(
        "surprise_kwargs",
        {"k": 50, "sim_options": {"name": "pearson_baseline", "user_based": False}},
    )
    # Check that the columns are
    assert set(important_cols) <= set(
        df.columns
    ), "Important cols are not a subset of the DataFrame"

    reader = Reader(rating_scale=rating_scale)

    data = Dataset.load_from_df(df[important_cols], reader)
    trainset, testset = train_test_split(data, test_size=test_size)

    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    algo = KNNWithMeans(**surprise_kwargs)
    algo.fit(trainset)

    # run the trained model against the testset
    test_pred = algo.test(testset)
    # run the trained model against the trainst
    train_pred = algo.test(trainset.build_testset())

    # get RMSE
    if verbose:
        rmse_test = accuracy.rmse(test_pred)
        print(f"Item-based Model : Test Set - {rmse_test}")

        rmse_train = accuracy.rmse(train_pred)
        print(f"Item-based Model : Training Set - {rmse_train}")

    return test_pred + train_pred


def load_baseline_cf(
    from_pickle: bool = True,
    drop_pickle: bool = False,
    pickle_file_name: str = None,
    df: pd.DataFrame = None,
    **kwargs,
):
    """
    Loads the Baseline CF Matrix from a pickle file or recalculates it.
    Args:
        from_pickle (bool): Defines whether the baseline CF Matrix should be loaded from the
                            pickle file or will be recalculated,

        drop_pickle (bool): If True then drop the current pickle file,

        pickle_file_name (None or str): If given then load pickle from this path name. Otherwise load from
                                        BASELINE_CF_PICKLE_FILE_NAME.
        df: (None or pandas.DataFrame): Needs to be passed when the baseline CF Matrix will be recalculated.
        **kwargs (dict): Dynamic dictionary with the initialization values of the 'calculate_baseline_cf'
                         function, in case of Baseline CF recalculation.

    Returns (pandas.DataFrame): A dataframe with the User - Item - Estimated Rating values.

    """

    def recalculate_baseline_cf():
        assert df is not None, "A DataFrame must be provided, in order to calculate the baseline CF Matrix."
        predictions = calculate_baseline_cf(df=df, **kwargs)
        return construct_cf_matrix(predictions)

    filepath = pickle_file_name or BASELINE_CF_PICKLE_FILE_NAME

    if drop_pickle:
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            print(e)

    if from_pickle:
        try:
            baseline_df = pd.read_pickle(filepath)
        except FileNotFoundError as e:
            baseline_df = recalculate_baseline_cf()
            baseline_df.to_pickle(filepath)
    else:
        baseline_df = recalculate_baseline_cf()

    return baseline_df
