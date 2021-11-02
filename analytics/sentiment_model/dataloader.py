import pandas as pd
import pickle
import datetime

batch_id = datetime.date.today().strftime("%y%m%d")

directory = {
    "statements": f"../data/db/pickle/extract/{batch_id}_statement.pickle",
    "minutes": f"../data/db/pickle/extract/{batch_id}_minutes.pickle",
    "news": f"../data/db/pickle/extract/{batch_id}_news.pickle",
    "model": "../data/db/pickle/model/model.pickle",
    "vectorizer": "../data/db/pickle/model/vectorizer.pickle",
    # historical dataframes: for updating
    "historical": {
        "statements": f"../data/db/pickle/historical/st_df.pickle",
        "minutes": f"../data/db/pickle/historical/mins_df.pickle",
        "news": f"../data/db/pickle/historical/news_df.pickle",
    },
}


class DataLoader:
    def __init__(self, start_dt):
        self.start_dt = start_dt
        self.data = self.load_data()

    def load_data(self):
        """
        Returns dictionary of dataframes
        """

        data = {}
        data["historical"] = {
            "statements": pd.DataFrame(),
            "minutes": pd.DataFrame(),
        }  # to set the structure

        for k, v in directory.items():
            if k == "historical":

                for k2, v2 in directory["historical"].items():
                    f = open(v2, "rb")
                    df = pickle.load(f)
                    data["historical"][k2] = df

            else:
                f = open(v, "rb")
                df = pickle.load(f)

                if k == "statements" or k == "minutes" or k == "news":
                    df = self.filter_date(df)

                data[k] = df

        return data

    def filter_date(self, df):
        """
        Filter only relevant dates based on users specification
        """
        # Ensure in datetime format
        df["date"] = pd.to_datetime(df["date"])

        # Filter based on from_year parameters
        start_date = pd.Timestamp(self.start_dt, 1, 1)
        df = df[(df["date"] >= start_date)]
        df.reset_index(inplace=True, drop=True)

        return df
