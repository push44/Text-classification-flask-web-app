import pandas as pd
from sklearn import model_selection


def create_folds(df):
    """
    :param df: Dataframe
    :return: Dataframe with added column indicating fold
    """
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.Y.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    return df


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    train = create_folds(train)
    train.to_csv("../input/train.csv", index=False)
