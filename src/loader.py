import pandas as pd
from .config import DATA_PATH

class DataLoader:
    def load(self) -> pd.DataFrame:
        if DATA_PATH.lower().endswith(".txt"):
            return pd.read_table(
                DATA_PATH, sep="\t", header=None,
                names=["cuenta","partido","timestamp","tweet"],
                encoding="latin-1",
                dtype=str
            )
        else:
            return pd.read_csv(DATA_PATH, encoding="latin-1", dtype=str)

    def save(self, df: pd.DataFrame):
        sep = "\t" if DATA_PATH.lower().endswith(".txt") else ","
        df.to_csv(DATA_PATH, sep=sep, index=False, encoding="latin-1")
