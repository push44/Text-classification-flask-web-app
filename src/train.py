import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import joblib


def Vectorize_tfidf(train, valid):
    """
    :param train: train predictors with dtype object
    :param valid: validation predictors
    :return: list train and validation stacked objects
    """
    train_tfidf = []
    valid_tfidf = []
    for feat in tqdm(train.columns):
        vec = TfidfVectorizer(ngram_range=(1, 4), max_features=10000)
        train_tfidf.append(vec.fit_transform(train[feat]))
        valid_tfidf.append(vec.transform(valid[feat]))

        joblib.dump(vec, f"../models/tfidf/{feat}_tfidf.pkl")

    train_stacked = sparse.hstack(train_tfidf).tocsr()
    valid_stacked = sparse.hstack(valid_tfidf).tocsr()

    return [train_stacked, valid_stacked]


def train_model(x_train, y_train):
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(x_train, y_train)

    return lr


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv").drop(["Id", "CreationDate", "body_text"], axis=1)
    valid = pd.read_csv("../input/valid.csv").drop(["Id", "CreationDate", "body_text"], axis=1)

    nrows = train.shape[0]

    df = pd.concat([train, valid])
    df.fillna("", inplace=True)

    df_y = df["Y"]

    df_y, uniques = pd.factorize(df_y, sort=True)

    df_x = df.drop(["Y"], axis=1)

    train_x = df_x.iloc[:nrows]
    valid_x = df_x.iloc[nrows:]
    train_y = df_y[:nrows]
    valid_y = df_y[nrows:]

    train_tfidf, valid_tfidf = Vectorize_tfidf(train_x.select_dtypes(include="object"),
                                               valid_x.select_dtypes(include="object"))

    train_x = sparse.hstack([train_x.select_dtypes(exclude="object").values, train_tfidf]).tocsr()

    valid_x = sparse.hstack([valid_x.select_dtypes(exclude="object").values, valid_tfidf]).tocsr()

    model = train_model(train_x, train_y)

    joblib.dump(model, "../models/LogisticRegression.pkl")
