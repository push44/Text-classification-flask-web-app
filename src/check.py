import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from train import create_padded_seq
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


def save_confusion_matrix(y_true, prediction):
    fig, ax = plt.subplots(figsize=(6, 6))
    confusion_matrix = metrics.confusion_matrix(y_true, prediction)
    sns.heatmap(confusion_matrix, annot=True, cbar=False, fmt=".1f", ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.savefig("../confusion_matrix.png")


def model_validation(df, model, tokenizer, max_len):
    corpus = df["clean_body_text"].values
    indicators = df[["code_indicator", "script_indicator"]].values
    targets = pd.get_dummies(df["Y"]).values

    seq = create_padded_seq(tokenizer, max_len, corpus)

    loss, acc = model.evaluate([seq, seq, seq, indicators], targets)
    print(f"Test loss:{loss}, Test Accuracy:{acc}")

    prediction = list(map(lambda val: np.argmax(val), model.predict([seq, seq, seq, indicators])))

    save_confusion_matrix(df["Y"].values, prediction)


if __name__ == "__main__":
    # Test data set
    test_dataframe = pd.read_csv("../input/test_dev.csv")
    # Trained CNN model
    trained_model = load_model("../models/cnn.h5")
    # Keras tokenizer
    with open("../models/keras_tokenizer.pkl", "rb") as f:
        keras_tokenizer = pickle.load(f)

    model_validation(test_dataframe, trained_model, keras_tokenizer, 2757)
