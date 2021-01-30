from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import sys

sys.path.insert(1, "./src/")
from dataframe import clean_body, code_indicator, script_indicator

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


model = load_model("models/cnn.h5")


def remove_punctuation(sentence):
    """
  Remove any punctuation marks and replace empty string with None
  """
    import re
    import string
    sentence = re.sub(f'[{re.escape(string.punctuation)}]', '', sentence)

    if sentence == "":
        sentence = "None"

    return sentence


def remove_numbers(sentence):
    """
  Remove any alpha-numeric or numeric words and replace empty string with None
  """
    sentence = " ".join([word for word in sentence.split() if word.isalpha()])

    if sentence == "":
        sentence = "None"

    return sentence


def remove_extra_spaces(sentence):
    """
  Remove all the extra spaces and replace empty string with None
  """
    sentence = sentence.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    sentence = sentence.replace("  ", " ").strip()

    if sentence == "":
        sentence = "None"

    return sentence


def create_padded_seq(tokenizer, max_length, sentences):
    """
    Convert all the sentences into sequences using keras tokenizer
    """
    from tensorflow import keras

    # create text to sequences
    seq = tokenizer.texts_to_sequences(sentences)

    # create actual sequences
    sequences = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding="post")

    return sequences


with open("models/keras_tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    _, body, _ = [x for x in request.form.values()]
    indicators = np.array([code_indicator(body), script_indicator(body)]).reshape(1, -1)
    body = remove_punctuation(body)
    body = remove_extra_spaces(body)
    body = remove_numbers(body)
    sentences = [body]

    max_len = 2757

    seq = create_padded_seq(tokenizer, max_len, sentences)
    seq = seq.reshape(1, -1)
    label_index = np.argmax(model.predict([seq, seq, seq, indicators])[0])

    labels = ['HQ', 'LQ_CLOSE', 'LQ_EDIT']
    labels_dict = {'HQ': "High Quality Question",
                   "LQ_CLOSE": "Low Quality Question Closed",
                   "LQ_EDIT": "Low Quality Question Edited"}

    output = labels_dict[labels[label_index]]

    return render_template("index.html", prediction_text="Question Label :{}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
