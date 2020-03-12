import os

import pandas as pd
from sklearn.metrics import pairwise_distances

from definitions import DATA_DIR

RATINGS_DATASET_PATH = os.path.join(DATA_DIR, "ratings.csv")

def get_items_of_category(category = -1, categorized_movies : pd.DataFrame = None):
    return categorized_movies[categorized_movies['Category'] == category]['Movie Id'].values

def get_suggestion_for_user(user = -1,movies: pd.DataFrame = None, L = 0):
    return movies.loc[user].sort_values(0,ascending = False).head(L)

def get_suggestion_for_all_users(movies: pd.DataFrame = None, L = 0):
    users = movies.index.unique()
    suggestions = dict()
    for user in users:
        suggestion = get_suggestion_for_user(user, movies, L)
        suggestions.update( {user : suggestion} )
    return suggestions

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
