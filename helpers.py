import itertools
import os
import random

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise
import pyomo.environ as pyo

from definitions import DATA_DIR

RATINGS_DATASET_PATH = os.path.join(DATA_DIR, "ratings.csv")


class MovieLensRatingsDataset(object):
    """
    Helper Class that loads the "ROOT_DIR/models/ratings.csv"
    file and expose a list operations over it, such as preprocessing and calculate similarities.
    Based on the initialization of the object the similarities can be user-based or item-based.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        df_path_name: str = None,
        preprocess_df: bool = True,
        user_based: bool = False,
        **kwargs,
    ):
        """
        If neither 'df' nor 'df_path_name' are given then the dataframe wille be loaded from the
        RATINGS_DATASET_PATH.
        Args:
            df (None or df.DataFrame): If given then the dataframe is not reloaded from file,
            df_path_name (None or path str): Describes a custom pathname,
            preprocess_df (bool): If True then preprocess the dataframe,
            user_based (bool): If True then the similarities will be User - User. Otherwise
                                they will be Item - Item.
            **kwargs:
        """

        self.drop_tsmpt = kwargs.get("drop_timestamp", True)
        self.df = (
            df
            if not (df is None or df.empty)
            else pd.read_csv(df_path_name or RATINGS_DATASET_PATH)
        )

        if preprocess_df:
            self.df = self.preprocess_dataframe()

        keep = kwargs.get("keep", 1.00)
        if keep < 1.00:
            self.df = self.as_subset(keep)

        self.pivot_df = self.df.pivot(
            index="movieId" if user_based else "userId",
            columns="userId" if user_based else "movieId",
            values="rating",
        ).fillna(0)

        # Create UsGurobiDirecter x Movies DataFrame from the pivoted DataFrame
        # It can be used for the Baseline CF matrix calculation
        pivot_df_cp = self.pivot_df.copy(deep=True)
        self.full_df = pivot_df_cp.unstack(
            "movieId" if user_based else "userId"
        ).reset_index()
        if user_based:
            self.full_df.columns = ["userId", "movieId", "rating"]
        else:
            self.full_df.columns = ["movieId", "userId", "rating"]

    def as_subset(self, keep: float = 1.00):
        """
        Based on the passed 'keep' argument return the relevant (keep*100)% of the dataset.
        The User-Movie removal will be equal and random.
        """
        import random

        assert (
            0.0 < keep <= 1.00
        ), "Invalid value of argument 'keep'. It should be 0.0 < 'keep' <= 1.00,"

        users = (
            self.df.groupby("userId")
            .mean()
            .sort_values(by=["rating"], ascending=False)
            .index.values.tolist()
        )
        movies = (
            self.df.groupby("movieId")
            .mean()
            .sort_values(by=["rating"], ascending=False)
            .index.values.tolist()
        )

        return self.df[
            self.df["userId"].isin(random.sample(users, int(len(users) * keep)))
            & self.df["movieId"].isin(random.sample(movies, int(len(movies) * keep)))
        ]

    def preprocess_dataframe(self, verbose=False):
        """
        Method that preprocess the loaded Dataframe. Steps of Preprocessing
        1. Remove Users that has less than 50 ratings
        2. Remove Movies that have movie_ratings_counter < mean_ratings_of_all_movies
        """

        if self.drop_tsmpt:
            self.df.drop(columns="timestamp")

        grouped_user_ratings = self.df.groupby("userId")
        if verbose:
            print(f"Starting data filtering shape{self.df.shape}")

        filtered_by_user = grouped_user_ratings.filter(lambda x: len(x["rating"]) >= 50)

        if verbose:
            print(f"1st data filtering shape{filtered_by_user.shape}")

        grouped_movie_ratings = filtered_by_user.groupby("movieId")
        mean_movies_ratings = grouped_movie_ratings.count()["rating"].mean()
        self.df = grouped_movie_ratings.filter(
            lambda x: len(x["rating"]) >= mean_movies_ratings
        )

        if verbose:
            print(f"2nd data filtering shape{self.df.shape}")

        return self.df

    def common_items_similarity(self, sim_method: str = "cosine"):
        """

        Args:
            sim_method:

        Returns:

        """
        col_dim = self.pivot_df.shape[1]
        similarity_pairs = []

        for elem1, elem2 in itertools.combinations(self.pivot_df.items(), 2):
            # This array would have all the indexes
            # of both elem1 and elem2 of the nonzero values
            elem1_val = elem1[1]
            elem1_idx = elem1[0]
            elem2_val = elem1[1]
            elem2_idx = elem2[0]

            common_choices = np.vstack((elem1_val, elem2_val)).T.nonzero()[0]
            common_indexes = [
                common_choices[i]
                for i in range(0, len(common_choices) - 1)
                if i + 1 < len(common_choices)
                and common_choices[i] == common_choices[i + 1]
            ]

            if sim_method == "cosine":
                import ipdb

                ipdb.set_trace()
                sim_val = pairwise.cosine_similarity(
                    np.vstack(
                        (elem1_val.iloc[common_indexes], elem2_val.iloc[common_indexes])
                    )
                )
                similarity_pairs += [
                    (
                        {"elem1": elem1_idx, "elem2": elem2_idx, "value": sim_val},
                        {"elem1": elem2_idx, "elem2": elem1_idx, "value": sim_val},
                    )
                ]

        sim_df = pd.DataFrame(similarity_pairs)
        pivot_sim_df = sim_df.pivot(
            index="elem1", columns="elem2", values="rating"
        ).fillna(0)
        np.fill_diagonal(pivot_sim_df.values, 1.00)

        return pivot_sim_df

    @property
    def jaccard_similarity(self):
        if not hasattr(self, "_jaccard_sim"):
            zero_one_df = self.pivot_df.copy(deep=True)
            zero_one_df[zero_one_df > 0.5] = 1
            zero_one_df = zero_one_df.astype(int)

            jac_sim_matrix = 1 - pairwise_distances(
                zero_one_df.values.T, metric="jaccard"
            )

            # optionally convert it to a DataFrame
            self._jaccard_sim = pd.DataFrame(
                jac_sim_matrix, index=zero_one_df.columns, columns=zero_one_df.columns
            )

        return self._jaccard_sim

    @property
    def pearson_similarity(self):
        if not hasattr(self, "_pearson_sim"):
            self._pearson_sim = self.pivot_df.corr(method="pearson")

        return self._pearson_sim

    @property
    def cosine_similarity(self):
        if not hasattr(self, "_cosine_sim"):
            cos_sim_matrix = 1 - pairwise_distances(
                self.pivot_df.values.T, metric="cosine"
            )
            # optionally convert it to a DataFrame
            self._cosine_sim = pd.DataFrame(
                cos_sim_matrix,
                index=self.pivot_df.columns,
                columns=self.pivot_df.columns,
            )
        return self._cosine_sim

    @property
    def hamming_similarity(self):
        if not hasattr(self, "_hamming_sim"):
            ham_sim_matrix = 1 - pairwise_distances(
                self.pivot_df.values.T, metric="hamming"
            )
            # optionally convert it to a DataFrame
            self._hamming_sim = pd.DataFrame(
                ham_sim_matrix,
                index=self.pivot_df.columns,
                columns=self.pivot_df.columns,
            )

        return self._hamming_sim


class CategoriesDataset(object):
    """
    A class object that creates a categories pd.Dataframe based on specific item_id list.
    Each category is generated randomly and each item belongs only to one category.
    """

    def __init__(
        self, items_ids: (np.ndarray, list, set), c: int = 5, **kwargs,
    ):

        assert isinstance(
            items_ids, (np.ndarray, list, set)
        ), "Argument 'items_ids' should be one of the following types (np.ndarray, list, set)"
        self.items_ids = items_ids
        self.num_c = c
        self.max_c = int(len(items_ids) / self.num_c) + 1

        self.categories_df = self.generate_categories_matrix()

    def generate_categories_matrix(self, refresh: bool = False):
        """
        TODO: Find better way to represent the categories. Due to optimizer formation
        inside the categories matrix we are not keeping the itemId but the movie Index.
        For example if itemId = 114 but it is in the index=5 position of list "items_ids"
        then the index will be added as value to the category matrix.
        Args:
            refresh:

        Returns:

        """
        categories = np.random.randint(
            low=1, high=self.num_c + 1, size=len(self.items_ids)
        )
        _, counts = np.unique(categories, return_counts=True)
        categories_df = pd.DataFrame(
            0, index=np.arange(self.num_c), columns=np.arange(max(counts))
        )

        zero_idxs = np.zeros(self.num_c)
        for idx, category_id in enumerate(categories):
            categories_df.iloc[category_id - 1, int(zero_idxs[category_id - 1])] = idx
            zero_idxs[category_id - 1] += 1

        return categories_df


class BaselineRIDataset(object):
    """
    A class object that filters the Top L items for each row of  a given pd.Dataframe
    which defines the result of the Collaborative Filtering Algorithm.
    """

    def __init__(self, df: pd.DataFrame, default_limit: int = 5):
        self.initial_baseline_rs = df
        self.default_suggestion_limit = default_limit

        if self.initial_baseline_rs.shape[1] > self.default_suggestion_limit:
            self.rec_df = self.suggestions()
        else:
            self.rec_df = self.initial_baseline_rs

    def _get_user_suggestion_list(self, user_id: int, limit: int = None):
        """
        Returns the pd.Series with the Top Limit Movies of a spesic 'user_id'
        Args:
            user_id (int): A valid index value of the dataframe
            limit (int or None): If given it defines the returned limit of the
        """
        limit = limit or self.default_suggestion_limit
        return (
            self.initial_baseline_rs.loc[user_id]
            .sort_values(0, ascending=False)
            .head(limit)
        )

    def suggestions(self, limit: int = None):
        """
        Filters the 'initial_baseline_rs' with the Top l movies with the higher rating values.
        Args:
            limit :

        Returns (pd.DataFrame): The formatting is the sane with the 'initial_baseline_rs'

        """
        limit = limit or self.default_suggestion_limit
        items_col_name = self.initial_baseline_rs.columns.name
        user_index_name = self.initial_baseline_rs.index.name

        users = self.initial_baseline_rs.index
        suggestions = []

        for user in users:
            limited_movies = self._get_user_suggestion_list(user, limit)
            for itemId, rating in zip(
                limited_movies.index.values.tolist(), limited_movies.values.tolist()
            ):
                suggestions.append(
                    {user_index_name: user, items_col_name: itemId, "rating": rating}
                )

        return (
            pd.DataFrame(suggestions)
            .pivot(index=user_index_name, columns=items_col_name, values="rating")
            .fillna(0)
        )


class ProblemAPredictionsMatrix(object):
    """
    Class Helper that takes the initial baseline prediction matrix and the prediction values
    of the Pyomo Variable and constructs a matrix with UxL domain and as values the recommended
    movie ids.
    """

    def __init__(self, bs_matrix: pd.DataFrame, x_var_pred: pyo.Var, L: int):
        self.bs_mat = bs_matrix
        self.x_var = x_var_pred
        self.l = L
        self.u = self.bs_mat.shape[0]
        self.i = self.bs_mat.shape[1]
        self.movie_ids = self.bs_mat.columns
        self.movies_predictions = self.construct_recommendation_matrix()
        self.predictions_average = self.calculate_predictions_average()

    def construct_recommendation_matrix(self):
        movies_predictions = {}

        for u in range(0, self.u):
            user_idx = self.bs_mat.iloc[u].name
            start = u * self.i
            # gets the user recommendation list and transforms
            # it to (movie_idx, x_pred_val) list of pairs

            u_pred = list(self.x_var.get_values().values())[start : start + self.i]
            u_pred_idx = [(idx, u_val) for idx, u_val in enumerate(u_pred)]
            # import ipdb
            # ipdb.set_trace()
            # create the L recommendation list for the user
            x_predictions = sorted(u_pred_idx, key=lambda x: x[1], reverse=True)[
                : self.l
            ]
            movies_predictions.update(
                {
                    user_idx: [
                        self.movie_ids[movie_idx] for movie_idx, _ in x_predictions
                    ]
                }
            )

        return movies_predictions

    def calculate_predictions_average(self):
        sum_r = 0
        for u_idx, movies in self.movies_predictions.items():
            sum_u_r = 0
            for m_idx in movies:
                sum_u_r += self.bs_mat.loc[(u_idx, m_idx)]
            sum_r += sum_u_r

        return sum_r / (self.u * self.l)
