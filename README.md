![Alt text](stackoverflow.png?raw=true "logo")

## Project Requiremnet

Stakoverflow is an widely used question answeres forum website that helps millions of users worldwide. But because of wide range of user base monitoring quality of the questions posted by the users becomes important. For example, when someone asks a question how to solve data science porblem? it becomes difficult for domain experts to answer the question when sufficient problem description is unavailble. Whereas when somebuddy asks how to solve multiclass text classification problem using logistic regression? it becomes much convenient for experts to answer the question.
In this project we tried to solve a such problem using the data set provided by [Moore](https://www.kaggle.com/imoore) on [Kaggle](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate)

## Web app preview
![Alt text](app.png?raw=true "Title")

### Please visit the app by clicking on the link below
[Stackoverflow text classification app](https://text-label-prediction.herokuapp.com/)

## File directory
```bash
.
├── app.py
├── input
│   ├── archive
│   │   ├── train.csv
│   │   └── test.csv
│   ├── train.csv
│   └── test.csv
├── models
│   ├── LogisticRegression.pkl
│   └── tfidf
│       ├── tags_tfidf.pkl
│       └── title_text_tfidf.pkl
├── nltk.txt
├── Procfile
├── requirements.txt
├── sample.docx
├── src
│   ├── create_folds.py
│   ├── predict.py
│   ├── __pycache__
│   │   └── test_text_features.cpython-38.pyc
│   ├── test_text_features.py
│   ├── text_features.py
│   └── train.py
├── static
│   └── css
│       └── style.css
└── templates
    └── index.html
```

## Techniques:
<b>Web development:</b> falsk, heorku cloud platform <br>
<b>Machine learning:</b> BeautifulSoup, nltk, re, TfIdfVectorizer, sklearn, logistic regression

## Machine learning problem formulation:<br>
The problem we are trying to solve is a *multi-class classification* problem.<br>
Classes:
<ul>
  <li>HQ: High Quality Question</li>
  <li>LQ_CLOSE: Low Quality Question Closed</li>
  <li>LQ_EDIT: Low Quality Question Edited</li>
</ul>

Raw data set columns:
<ul>
  <li>Id</li>
  <li>Title</li>
  <li>Body</li>
  <li>Tags</li>
  <li>Creation Date</li>
</ul>

Hence, this is a *text multi-class classification* problem.

## Methodology:
<ol>
  <li><b>create_folds.py</b>: Creates a column called "kfold" that assigns each data row one of the 10 fold value.</li>
  <li><b>text_features.py</b>: This is a second step for any text related machine learning problem, where we clean up all the text.
     <ul>
      <li>Remove all tags "<", ">", and other punctuation marks.</li>
      <li>Remove all the stop words</li>
      <li>Remove all code snippets</li>
      <li>Remove all reference links</li>
      <li>Remove all images</li>
      <li>Lowercase all words</li>
    </ul>
  
  Then we also create three different features:
    <ul>
      <li>code_indicator: Binary feature indicates if text contains a code snippet</li>
      <li>reference_link_indicator: Binary feature indicates if text contains a reference link</li>
      <li>image_link_indicator: Binary feature indicates if text contains an image</li>
    </ul>

  </li>
  <li><b>test_text_features.py</b>: This file essentially does the same job as text_features.py but instead intended to operate only on the test text provided through the app.py file</li>
  <li><b>train.py</b>: As name suggests, we train out logistic regression model in this file and use joblib to dump model in the models folder.</li>
  <li><b>predict.py</b>: This file is used to make final prediction.</li>
</ol>

## Flask app:
Once the model is built we can create our intended Flask web app that will clasify given question into one of the three categories. This is done through the <b>app.py</b> file.

## Model Deployment:
We used Heroku Cloud Platform that is easy to use and is free for hosting small web apps like this.

## Thank you visting!
