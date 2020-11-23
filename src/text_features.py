import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tqdm import tqdm
import re


def clean_tags(string):
    """
    :param string: Takes single string
    :return: return string with tags removed
    """
    return ((string.replace('><', ' ')).replace('<', '')).replace('>', '')


def clean_body_text(df):
    """
    :param df: Take dataframe as input
    :return: Extracted features in list formats
    """
    # Create list of english stopwords from nltk library
    stop_words = set(stopwords.words('english'))

    # Create a list to save body text of all questions
    body_text = []
    # Create a list to indicate if code snippet is present in the body
    code_indicator = []
    reference_link_indicator = []
    image_indicator = []

    for ind in tqdm(range(df.shape[0])):

        # Create a BeautifulSoup object
        q_body = df['Body'].values[ind].lower()
        soup = BeautifulSoup(q_body)

        # To check if body contains code snippet
        if len(soup.findAll('code')) > 0:
            code_indicator.append(1)
            # Find all code tags and replace them with empty string ''
            for code_text in soup.findAll('code'):
                code_text.replace_with('')
        else:
            code_indicator.append(0)

        # To check if body contains reference link tag
        if len(soup.findAll('a')) > 0:
            reference_link_indicator.append(1)
        else:
            reference_link_indicator.append(0)

        # To check if body contains image
        if len(soup.findAll('img')) > 0:
            image_indicator.append(1)
        else:
            image_indicator.append(0)

            # Create a list to save all <p> tag text of a question into a list
        text = []
        for line in soup.findAll('p'):
            line = line.get_text()
            line = line.replace('\n', '')
            line = re.sub(r'[^A-Za-z0-9]', ' ', line)
            line = ' '.join([word for word in line.split() if not word in stop_words])
            text.append(line)

        body_text.append(' '.join(text))

    return [body_text, code_indicator, reference_link_indicator, image_indicator]


def clean_title_text(df):
    # Create list of english stopwords from nltk library
    stop_words = set(stopwords.words('english'))
    title_text = []
    for ind in range(df.shape[0]):
        text = df.Title.values[ind].lower()
        text = text.replace('\n', '')
        text = re.sub(r'[^A-Za-z0-9]', ' ', text)
        text = ' '.join([word for word in text.split() if not word in stop_words])

        title_text.append(text)

    return title_text


def run(train, valid):

    n_rows = train.shape[0]
    df = pd.concat([train, valid], axis=0, ignore_index=True)

    df['tags'] = list(map(lambda val: clean_tags(val), df.Tags.values))
    df.drop(['Tags'], axis=1, inplace=True)

    df['body_text'], df['code_indicator'], df['reference_link_indicator'], df['image_indicator'] =\
        clean_body_text(df)

    df['title_text'] = clean_title_text(df)

    df = df.drop(['Title', 'Body'], axis=1)

    train = df.iloc[:n_rows]
    valid = df.iloc[n_rows:]

    return [train, valid]


if __name__ == "__main__":
    df_train = pd.read_csv("../input/train.csv")
    df_valid = pd.read_csv("../input/valid.csv")
    df_train, df_valid = run(df_train, df_valid)

    df_train.to_csv("../input/train.csv", index=False)
    df_valid.to_csv("../input/valid.csv", index=False)
