import itertools
import os
import random

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise

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
        self.df = df if df else pd.read_csv(df_path_name or RATINGS_DATASET_PATH)

        if preprocess_df:
            self.df = self.preprocess_dataframe()

        self.pivot_df = self.df.pivot(
            index="movieId" if user_based else "userId",
            columns="userId" if user_based else "movieId",
            values="rating",
        ).fillna(0)

        # Create User x Movies DataFrame from the pivoted DataFrame
        # It can be used for the Baseline CF matrix calculation
        pivot_df_cp = self.pivot_df.copy(deep=True)
        self.full_df = pivot_df_cp.unstack(
            "userId" if user_based else "movieId"
        ).reset_index()
        if user_based:
            self.full_df.columns = ["userId", "movieId", "rating"]
        else:
            self.full_df.columns = ["movieId", "userId", "rating"]

    def preprocess_dataframe(self, verbose=False):
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
                    np.vstack((elem1_val.iloc[common_indexes], elem2_val.iloc[common_indexes]))
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

    """

    def __init__(
        self,
        items_ids: (np.ndarray, list, set),
        c: int = 5,
        shuffle: bool = True,
        **kwargs,
    ):

        assert isinstance(
            items_ids, (np.ndarray, list, set)
        ), "Argument 'items_ids' should be one of the following types (np.ndarray, list, set)"
        self.items_ids = items_ids
        self.num_c = c
        self.shuffle = shuffle

        self.categories_df = self.generate_categories_matrix()

    def generate_categories_matrix(self, refresh: bool = False):
        if self.shuffle:
            random.shuffle(self.items_ids)

        categories = np.random.randint(low=1, high=self.num_c, size=len(self.items_ids))
        if not hasattr(self, "categories_df") or refresh:
            self.categories_df = pd.DataFrame(
                {
                    "movieId": self.items_ids[:],
                    "category": categories[:],
                    "value": np.ones(len(self.items_ids)),
                }
            )
        return self.categories_df

    @property
    def pivot_categories(self):
        if not hasattr(self, "_pivot_df"):
            self._pivot_df = (
                self.categories_df.pivot(
                    index="category", columns="movieId", values="value"
                )
                .fillna(0)
                .astype(int)
            )
        return self._pivot_df


class BaselineRSDataset(object):
    """

    """

    def __init__(self, df: pd.DataFrame, l: int = 5, **kwargs):
        self.baseline_rs = df
        self.suggestion_num = l

    def _get_user_suggestion_list(self, user_id):

        return (
            self.baseline_rs.loc[user_id]
            .sort_values(0, ascending=False)
            .head(self.suggestion_num)
        )

    @property
    def suggestions(self):
        users = self.baseline_rs.index.unique()
        if not hasattr(self, "_suggestions"):
            self._suggestions = {
                user_id: self._get_user_suggestion_list(user_id) for user_id in users
            }
        return self._suggestions
