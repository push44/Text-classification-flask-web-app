from flask import Flask, render_template, request
import joblib

import sys
sys.path.insert(1, "./src/")
from test_text_features import clean_tags, clean_body_text, clean_title_text

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
    body_text, code_indicator, reference_link_indicator, image_indicator = clean_body_text(body)
    title_text = clean_title_text(title)

    predictor = []
    features = dict(zip("tags code_indicator reference_link_indicator image_indicator title_text".split(),
                    [tags, code_indicator, reference_link_indicator, image_indicator, title_text]))
    for feat in features.keys():
        try:
            vec = joblib.load(f"./models/tfidf/{feat}_tfidf.pkl")
            predictor.append(vec.transform([features[feat]]))

        except:
            predictor.append(int(features[feat]))

    X = sparse.hstack(predictor).tocsr()

    labels = ['HQ', 'LQ_CLOSE', 'LQ_EDIT']
    labels_dict = {'HQ': "High Quality Question",
                   "LQ_CLOSE": "Low Quality Question Closed",
                   "LQ_EDIT": "Low Quality Question Edited"}

    output = labels_dict[labels[model.predict(X)[0]]]

    return render_template("index.html", prediction_text="Question Label :{}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
