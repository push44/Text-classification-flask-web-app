from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re


def clean_tags(string):
    """
    :param string: Takes single string
    :return: return string with tags removed
    """
    return ((string.replace('><', ' ')).replace('<', '')).replace('>', '')


def clean_body_text(body):
    """
    :param body: string of body
    :return: Extracted features in list formats
    """
    # Create list of english stopwords from nltk library
    stop_words = set(stopwords.words('english'))

    # Create a BeautifulSoup object
    q_body = body.lower()
    soup = BeautifulSoup(q_body)

    # Create a list to save all <p> tag text of a question into a list
    text = []
    for line in soup.findAll('p'):
        line = line.get_text()
        line = line.replace('\n', '')
        line = re.sub(r'[^A-Za-z0-9]', ' ', line)
        line = ' '.join([word for word in line.split() if not word in stop_words])
        text.append(line)

    body_text = ' '.join(text)

    # To check if body contains code snippet
    if len(soup.findAll('code')) > 0:
        code_indicator = 1
        # Find all code tags and replace them with empty string ''
        for code_text in soup.findAll('code'):
            code_text.replace_with('')
    else:
        code_indicator = 0

    # To check if body contains reference link tag
    if len(soup.findAll('a')) > 0:
        reference_link_indicator = 1
    else:
        reference_link_indicator = 0

    # To check if body contains image
    if len(soup.findAll('img')) > 0:
        image_indicator = 1
    else:
        image_indicator = 0

    return [body_text, code_indicator, reference_link_indicator, image_indicator]


def clean_title_text(title_string):
    """
    :param title_string: title in string format
    :return: preprocessed title text
    """
    # Create list of english stopwords from nltk library
    stop_words = set(stopwords.words('english'))

    text = title_string.lower()
    text = text.replace('\n', '')
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    text = ' '.join([word for word in text.split() if not word in stop_words])

    return text
