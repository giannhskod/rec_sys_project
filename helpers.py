import os

import pandas as pd

from definitions import DATA_DIR

RATINGS_DATASET_PATH = os.path.join(DATA_DIR, "ratings.csv")


class PreprocessRatingsDataset(object):
    """

    """
    def __init__(self, df: pd.DataFrame = None, df_path_name: str = None, **kwargs):
        """

        Args:
            df:
            df_path_name:
            **kwargs:
        """
        assert df is not None or df_path_name, f"One of the two parameters 'df' or 'df_path_name' should be passed."

        self.drop_tsmpt = kwargs.get("drop_timestamp", True)
        
        if not df:

            self.df = pd.read_csv(RATINGS_DATASET_PATH)
        else:
            self.df = df

        if self.drop_tsmpt:
            self.df.drop(columns="timestamp")

    def preprocess_dataframe(self, verbose=False):
        grouped_user_ratings = self.df.groupby("userId")
        if verbose:
            print(f"Starting data filtering shape{self.df.shape}")

        filtered_by_user = grouped_user_ratings.filter(lambda x: len(x["rating"]) >= 50)

        if verbose:
            print(f"1st data filtering shape{filtered_by_user.shape}")

        grouped_movie_ratings = filtered_by_user.groupby("movieId")
        mean_movies_ratings = grouped_movie_ratings.count()["rating"].mean()
        self.df = grouped_movie_ratings.filter(lambda x: len(x["rating"]) >= mean_movies_ratings)

        if verbose:
            print(f"2nd data filtering shape{self.df.shape}")

        return self.df



