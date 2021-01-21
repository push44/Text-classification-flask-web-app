import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import tqdm
from scipy import sparse
import joblib
import nltk

nltk.download("punkt")


def remove_extra_spaces(sentence):
    sentence = sentence.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    sentence = sentence.replace("  ", " ").strip()

    return sentence


def remove_punctuation(sentence):
    import re
    import string
    sentence = re.sub(f'[{re.escape(string.punctuation)}]', '', sentence)

    if sentence == "":
        sentence = "None"

    return sentence


def remove_numbers(sentence):
    sentence = " ".join([char for char in sentence.split() if not char.isdigit()])

    if sentence == "":
        sentence = "None"

    return sentence


def model_train(train_df, test_df):
    xtrain = []
    xtest = []

    features = ["Title", "Body", "Tags"]
    for feat in tqdm(features):
        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, lowercase=True, token_pattern="", max_df=0.8,
                                    max_features=10000)

        tfidf_vec.fit(train_df[feat])

        joblib.dump(tfidf_vec, f"../models/tfidf/{feat}_tfidf.pkl")

        xtrain.append(tfidf_vec.transform(train_df[feat]))
        xtest.append(tfidf_vec.transform(test_df[feat]))

    xtrain.append(train_df["code_indicator"].values.reshape(-1, 1))
    xtest.append(test_df["code_indicator"].values.reshape(-1, 1))

    xtrain = sparse.hstack(xtrain)
    xtest = sparse.hstack(xtest)

    print(f"Training model...")
    model = LogisticRegression(max_iter=10000)

    model.fit(xtrain, train_df["Y"])
    joblib.dump(model, "../models/LogisticRegression.pkl")
    print("Test Score:", round(metrics.f1_score(test_df["Y"], model.predict(xtest), average="micro"), 4))


if __name__ == "__main__":
    train = pd.read_csv("../input/train_dev.csv")
    test = pd.read_csv("../input/test_dev.csv")

    train["clean_body_text"] = train["clean_body_text"].apply(remove_punctuation)
    train["clean_body_text"] = train["clean_body_text"].apply(remove_extra_spaces)
    train["clean_body_text"] = train["clean_body_text"].apply(remove_numbers)

    test["clean_body_text"] = test["clean_body_text"].apply(remove_punctuation)
    test["clean_body_text"] = test["clean_body_text"].apply(remove_extra_spaces)
    test["clean_body_text"] = test["clean_body_text"].apply(remove_numbers)

    model_train(train, test)
