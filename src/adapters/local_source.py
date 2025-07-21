import pandas as pd


class LocalDataSource:
    def __init__(self, pth):
        self.pth = pth

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.pth)
