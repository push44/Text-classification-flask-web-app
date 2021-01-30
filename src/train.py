import pandas as pd
import numpy as np


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


def create_tokenizer(sentences):
    """
  Fit keras tokenizer on he corpus and return that tokenizer
  """
    from tensorflow import keras

    # initialize tokenizers of necessary specifications
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=999)
    # use tokenizer to on the input corpus of sentences
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def create_padded_seq(tokenizer, max_length, sentences):
    """
    Convert all the sentences into sequences using keras tokenizer
    """
    from tensorflow import keras

    # create text to sequences
    corpus = tokenizer.texts_to_sequences(sentences)

    # create actual sequences
    padded_corpus = keras.preprocessing.sequence.pad_sequences(corpus, maxlen=max_length, padding="post")

    return padded_corpus


def create_model(length, vocab_size):
    # https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
    from tensorflow import keras
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    # channel 1
    inputs1 = keras.layers.Input(shape=(length,))
    embedding1 = keras.layers.Embedding(vocab_size, 100)(inputs1)
    conv1 = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = keras.layers.Dropout(0.5)(conv1)
    pool1 = keras.layers.MaxPooling1D(pool_size=2)(drop1)
    flat1 = keras.layers.Flatten()(pool1)

    # channel 2
    inputs2 = keras.layers.Input(shape=(length,))
    embedding2 = keras.layers.Embedding(vocab_size, 100)(inputs2)
    conv2 = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding2)
    drop2 = keras.layers.Dropout(0.5)(conv2)
    pool2 = keras.layers.MaxPooling1D(pool_size=2)(drop2)
    flat2 = keras.layers.Flatten()(pool2)

    # channel 3
    inputs3 = keras.layers.Input(shape=(length,))
    embedding3 = keras.layers.Embedding(vocab_size, 100)(inputs3)
    conv3 = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding3)
    drop3 = keras.layers.Dropout(0.5)(conv3)
    pool3 = keras.layers.MaxPooling1D(pool_size=2)(drop3)
    flat3 = keras.layers.Flatten()(pool3)

    # channel 4
    inputs4 = keras.layers.Input(shape=(2,))

    # merge
    merged = keras.layers.concatenate([flat1, flat2, flat3, inputs4])

    # interpretation
    dense1 = keras.layers.Dense(10, activation='relu')(merged)
    outputs = keras.layers.Dense(3, activation='sigmoid')(dense1)
    model = keras.models.Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)

    # compile
    model.compile(loss='CategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])

    # keras.utils.plot_model(model, show_shapes=True)
    return model


def text_preparation(df):
    # Apply above defined text preprocessing function
    df.loc[:, "clean_body_text"] = df["clean_body_text"].apply(remove_punctuation)
    df.loc[:, "clean_body_text"] = df["clean_body_text"].apply(remove_extra_spaces)
    df.loc[:, "clean_body_text"] = df["clean_body_text"].apply(remove_numbers)
    # Create padded sequences from cleaned raw text
    corpus = df["clean_body_text"].values
    return corpus


def model_train(df):
    import pickle

    # Get sentences
    sentences = text_preparation(df)
    # Get tokenizer
    tokenizer = create_tokenizer(sentences)
    # Find max length of sentence in train data
    max_len = np.max([len(train_line.split()) for train_line in sentences])

    # Save tokenizer
    with open("../models/keras_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)

    # Create sequences from sentences and do padding to them to maintain the shape uniformity for the CNN model.
    seq = create_padded_seq(tokenizer, max_len, sentences)

    # indicators
    indicators = df[["code_indicator", "script_indicator"]].values

    # Target values
    targets = pd.get_dummies(df["Y"]).values

    # Initialize the CNN model
    model = create_model(max_len, vocab_size=10000)

    # Fit the model to train data
    model.fit([seq, seq, seq, indicators], targets, epochs=2, batch_size=16)

    # save model
    model.save("../models/cnn.h5")

    return max_len


def model_validate(df, max_len):
    import pickle
    from tensorflow.keras.models import load_model
    # Max length
    print(f"Max length: {max_len}")
    # Get sentences
    sentences = text_preparation(df)

    # Load tokenizer
    with open("../models/keras_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    # Create sequences from sentences using pre-trained tokenizer and pre-calculated max length.
    seq = create_padded_seq(tokenizer, max_len, sentences)

    # indicators
    indicators = df[["code_indicator", "script_indicator"]].values

    # Target values
    targets = pd.get_dummies(df["Y"]).values

    # Load model
    model = load_model("../models/cnn.h5")

    # Evaluate model performance (validation performance).
    loss, acc = model.evaluate([seq, seq, seq, indicators], targets)
    print(f"Validation loss: {loss}, Validation accuracy: {acc}")


if __name__ == "__main__":

    # read train data
    train_data = pd.read_csv("../input/train_dev.csv")

    # read validation data
    test_data = pd.read_csv("../input/test_dev.csv")

    m_len = model_train(train_data)
    model_validate(test_data, m_len)
