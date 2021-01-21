from flask import Flask, render_template, request
import joblib

import sys

sys.path.insert(1, "./src/")
from dataframe import clean_tags, clean_body, code_indicator
# from test_text_features import clean_tags, clean_body_text, clean_title_text

from scipy import sparse

app = Flask(__name__)

model = joblib.load("./models/LogisticRegression.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    title, body, tags = [x for x in request.form.values()]

    tags = clean_tags(tags)
    indicator = code_indicator(body)
    body = clean_body(body)

    predictor = []

    title_vec = joblib.load("./models/tfidf/Title_tfidf.pkl")
    predictor.append(title_vec.transform([title]))

    body_vec = joblib.load("./models/tfidf/Body_tfidf.pkl")
    predictor.append(body_vec.transform([body]))

    tags_vec = joblib.load("./models/tfidf/Tags_tfidf.pkl")
    predictor.append(tags_vec.transform([tags]))

    predictor.append([indicator])

    X = sparse.hstack(predictor).tocsr()

    labels = ['HQ', 'LQ_CLOSE', 'LQ_EDIT']
    labels_dict = {'HQ': "High Quality Question",
                   "LQ_CLOSE": "Low Quality Question Closed",
                   "LQ_EDIT": "Low Quality Question Edited"}

    output = labels_dict[labels[model.predict(X)[0]]]

    return render_template("index.html", prediction_text="Question Label :{}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
