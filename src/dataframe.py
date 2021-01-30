import pandas as pd
from bs4 import BeautifulSoup
import nltk

nltk.download("wordnet")


def code_indicator(sentence):
    soup = BeautifulSoup(sentence.lower(), "lxml")
    # To check if body contains code snippet
    if len(soup.findAll('code')) > 0:
        return 1
    else:
        return 0


def script_indicator(sentence):
    soup = BeautifulSoup(sentence.lower(), "lxml")
    # To check if body contains code snippet
    if len(soup.findAll('script')) > 0:
        return 1
    else:
        return 0


def clean_tags(sentence):
    sentence = sentence.replace("<", " ").replace(">", " ").replace("><", " ").strip()
    sentence = sentence.replace("  ", " ")
    return sentence


def decontraction(sentence):
    CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    sentence_tokens = sentence.split()
    for ind, word in enumerate(sentence_tokens):
        if word in CONTRACTION_MAP.keys():
            sentence_tokens[ind] = CONTRACTION_MAP[word]
    sentence = " ".join(sentence_tokens)

    return sentence


def clean_body(sentence):
    # Create Soup object
    soup = BeautifulSoup(sentence, "lxml")
    # Find only p tags
    soup_content = [line.text for line in soup.findAll("p")]

    # Create string from list of strings and remove extra spaces
    string_sentences = " ".join(soup_content)
    string_sentences = string_sentences.replace("  ", " ").strip().lower()

    if len(string_sentences) > 0:
        pass
    else:
        string_sentences = "None"

    string_sentences = decontraction(string_sentences)

    return string_sentences


def code_len(sentence):
    soup = BeautifulSoup(sentence, "lxml")
    soup_content = [line.text for line in soup.findAll("code")]
    string = " ".join(soup_content).strip()
    return len(string)


def update_dataframe(df):
    # Convert target to numericals
    df["Y"], uniques = pd.factorize(df["Y"], sort=True)

    # Clean tags
    df["Tags"] = df["Tags"].apply(clean_tags)

    # Extract indicator features from body text
    # code_indicator, reference_link_indicator, image_indicator = body_text_indicator_features(df)
    df["code_indicator"] = df["Body"].apply(code_indicator)

    df["script_indicator"] = df["Body"].apply(script_indicator)

    # Create clean body text
    df["clean_body_text"] = df["Body"].apply(clean_body)

    # Extract code length
    df["code_length"] = df["Body"].apply(code_len)

    return df


if __name__ == "__main__":
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/valid.csv")

    train_df = update_dataframe(train_df)
    test_df = update_dataframe(test_df)

    train_df.to_csv("../input/train_dev.csv", index=False)
    test_df.to_csv("../input/test_dev.csv", index=False)
