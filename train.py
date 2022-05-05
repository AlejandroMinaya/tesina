#!/usr/bin/env python
"""
Training the CRF model
Tesina
Juan Alejandro Alcántara Minaya
Feb - Jun 2022
"""
from collections import Counter
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from sys import stdout
from time import process_time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import joblib
import logging as _logging
import numpy as np
import pandas as pd
import random
import nltk
import sklearn_crfsuite
import warnings
import re

CURR_YEAR = date.today().year
FULL_MONTHS = [
    "january", "february", "march",
    "april", "may", "june", "july",
    "august", "september", "october",
    "november", "december"
]
ABBR_MONTHS = [
    "jan", "feb", "mar",
    "apr", "may", "jun", "jul",
    "aug", "sep", "oct",
    "nov", "dec"
]
STOP_WORDS = set(stopwords.words("english"))

def print_transitions(trans_features):
    [
        print(f"{_from} -> {_to}: {wght}")\
        for (_from, _to), wght in trans_features
    ]


def print_state_features(state_features):
    [
        print(f"{wght} {label} {attr}")\
        for (attr, label), wght in state_features
    ]


def docs_to_features(ids, df):
    docs = []
    for idx in ids:
        doc_df = df[df["doc_id"] == idx]
        docs.append(list(doc_df.iloc[:,1]))
    features = []
    for doc in docs:
        features.append([
            word_to_features(doc, i)\
            for i in range(len(doc))
        ])
    return features


def docs_to_labels(ids, df):
    labels = []
    for idx in ids:
        doc_df = df[df["doc_id"] == idx]
        labels.append([str(x) for x in doc_df.iloc[:,2]])
    return labels

def syllable_count(word):
    count = 0
    for w in word:
        if w in ['a','e','i','o','u']:
            count += 1
    return count

def word_to_features(doc, idx):
    word = str(doc[idx])
    word_lower = word.lower()
    features = {
        "bias": 1.0,
        "word.istitle":  word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.is_year": (word.isdigit() and int(word) >= 1900),
        "word.len": len(word),
        "word.is_abbr_month": (word_lower in ABBR_MONTHS),
        "word.is_full_month": (word_lower in FULL_MONTHS),
        "word.syllable_count": syllable_count(word_lower),
        "word.has_comma": ("," in word),
        "word.has_dash": ("-" in word),
        "word.has_slash": ("/" in word),
        "word.last_close_bracket": (")" in word[-1]),
        "word.first_open_bracket": ("(" == word[0])
    }
    if word not in STOP_WORDS:
        features.update({
            "word.pos_tag": nltk.pos_tag(word_tokenize(word))[0][1]
        })
    if idx > 0:
        prev_word = str(doc[idx-1])
        prev_word_lower = prev_word.lower()
        features.update({
            "-1:word.istitle": prev_word.istitle(),
            "-1:word.is_year": (prev_word.isdigit() and int(prev_word) >= 1900),
            "-1:word.is_abbr_month": (prev_word_lower in ABBR_MONTHS),
            "-1:word.is_full_month": (prev_word_lower in FULL_MONTHS),
            "-1:word.isdigit": prev_word.isdigit(),
            "-1:word.syllable_count": syllable_count(prev_word_lower),
            "-1:word.has_comma": ("," in prev_word),
            "-1:word.has_dash": ("-" in prev_word),
            "-1:word.has_slash": ("/" in prev_word)
        })
        if prev_word not in STOP_WORDS:
            features.update({
                "-1:word.pos_tag": nltk.pos_tag(word_tokenize(prev_word))[0][1]
            })
    if idx < len(doc) - 1:
        next_word = str(doc[idx+1])
        next_word_lower = next_word.lower()
        features.update({
            "+1:word.istitle": next_word.istitle(),
            "+1:word.is_year": (next_word.isdigit() and int(next_word) >= 1900),
            "+1:word.is_abbr_month": (next_word_lower in ABBR_MONTHS),
            "+1:word.is_full_month": (next_word_lower in FULL_MONTHS),
            "+1:word.isdigit": next_word.isdigit(),
            "+1:word.syllable_count": syllable_count(next_word_lower),
            "+1:word.has_comma": ("," in next_word),
            "+1:word.has_dash": ("-" in next_word),
            "+1:word.has_slash": ("/" in next_word)
        })
        if next_word not in STOP_WORDS:
            features.update({
                "+1:word.pos_tag": nltk.pos_tag(word_tokenize(next_word))[0][1]
            })
    return features


if __name__ == "__main__":
    # Dowload nltk kit
    nltk.download("punkt")
    # Turn off warnings
    warnings.filterwarnings("error")
    # Logger set-up
    logging = _logging.getLogger()
    logging.setLevel(0)
    handler = _logging.StreamHandler(stdout)
    handler.setLevel(0)
    handler.setFormatter(_logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s"
    ))
    logging.addHandler(handler)

    #data_path = input("Enter the data path (CSV): ")
    data_path = "output.csv"
    df = pd.read_csv(data_path, header=0)
    doc_ids = list(df["doc_id"].unique())
    random.shuffle(doc_ids)

    x_raw_train_ids, x_raw_test_ids, _, _ =\
        train_test_split(doc_ids, [0]*len(doc_ids), test_size=.2)
    x_train = docs_to_features(x_raw_train_ids, df)
    y_train = docs_to_labels(x_raw_train_ids, df)
    x_test = docs_to_features(x_raw_test_ids, df)
    y_test = docs_to_labels(x_raw_test_ids, df)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.05,
        c2=0.05,
        max_iterations=300,
        all_possible_transitions=True
    )
    train_start = process_time()
    crf.fit(x_train, y_train)
    logging.info(f"CRF training: {process_time() - train_start}s")
    joblib.dump(crf, "model.pkl", compress=9)

    # Reports
    y_pred = crf.predict(x_test)
    print(sorted(crf.classes_))
    flat_y_test = []
    for y_array in y_test:
        for y in y_array:
            flat_y_test.append(y)
    flat_y_pred = []
    for y_array in y_pred:
        for y in y_array:
            flat_y_pred.append(y)

    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted(np.unique(flat_y_pred))
    ))
    trans_features = Counter(crf.transition_features_)
    print("Top most likely transitions")
    print_transitions(trans_features.most_common(20))
    print("\nTop most unlikely transitions")
    print_transitions(trans_features.most_common()[-20:])
    state_features = Counter(crf.state_features_)
    print("\nTop positive features:")
    print_state_features(state_features.most_common(20))
    print("\nTop negative features:")
    print_state_features(state_features.most_common()[-20:])
