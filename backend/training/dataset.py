from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_jigsaw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["comment_text"] + LABEL_COLUMNS].dropna()
    # Ensure labels are int (0 or 1)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].astype(int)
    return df


def split_dataset(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


def dataframe_to_hf_dataset(
    df: pd.DataFrame,
) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def get_label_columns() -> List[str]:
    return LABEL_COLUMNS
