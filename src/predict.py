import pandas as pd
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse


def model_predict(model, x_train, x_valid):

    train = []
    valid = []

    for feat in x_train.columns:
        if feat in x_train.select_dtypes(include="object").columns:
            vec = joblib.load(f"../models/tfidf/{feat}_tfidf.pkl")
            train.append(vec.transform(x_train[feat]))
            valid.append(vec.transform(x_valid[feat]))

        else:
            train.append(x_train[feat].values.reshape(-1, 1))
            valid.append(x_valid[feat].values.reshape(-1, 1))


    x_train = sparse.hstack(train).tocsr()
    x_valid = sparse.hstack(valid).tocsr()

    training_prediction = model.predict(x_train)
    validation_prediction = model.predict(x_valid)

    return training_prediction, validation_prediction


def visualize(y_train, y_valid, y_train_pred, y_valid_pred):

    true_label = [y_train, y_valid]
    predict_label = [y_train_pred, y_valid_pred]
    sets = ["Train", "Validation"]

    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=2)

    for ind in range(len(ax)):

        print(f'{sets[ind]} mean accuracy score:', metrics.accuracy_score(true_label[ind], predict_label[ind]))

        sns.heatmap(metrics.confusion_matrix(true_label[ind], predict_label[ind]), annot=True, cbar=False, fmt='d',
                    cmap='Reds', ax=ax[ind])

        ax[ind].set_ylabel('True label', fontsize=14)
        ax[ind].set_xlabel('Predicted label', fontsize=14)
        ax[ind].set_title(f'Confusion matrix: {sets[ind]} set prediction', fontsize=16)

    plt.show()


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv").drop(["Id", "CreationDate", "body_text"], axis=1)
    valid = pd.read_csv("../input/valid.csv").drop(["Id", "CreationDate", "body_text"], axis=1)

    train.fillna("", inplace=True)
    valid.fillna("", inplace=True)

    train_y, valid_y = train["Y"], valid["Y"]

    train_y, _ = pd.factorize(train_y)
    valid_y, _ = pd.factorize(valid_y)

    train_x, valid_x = train.drop(["Y"], axis=1), valid.drop(["Y"], axis=1)

    logistic_reg = joblib.load("../models/LogisticRegression.pkl")

    train_predict, valid_predict = model_predict(logistic_reg, train_x, valid_x)
    visualize(train_y, valid_y, train_predict, valid_predict)