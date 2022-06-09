#!/usr/bin/env python
"""
Training the CRF model
Tesina
Juan Alejandro Alcántara Minaya
Feb - Jun 2022
"""
# import nltk
from collections import Counter
from datetime import date
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn_crfsuite import metrics
from sys import stdout
from time import process_time
import csv
import joblib
import logging as _logging
import numpy as np
import pandas as pd
import random
import re
import scipy
from sklearn_crfsuite import CRF
import warnings

FULL_MONTHS = [
    "enero", "febrero", "marzo",
    "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "octubre",
    "noviembre", "diciembre"
]
ABBR_MONTHS = [
    "ene", "feb", "mar",
    "abr", "may", "jun", "jul",
    "ago", "sep", "oct",
    "nov", "dec"
]
## FULL_MONTHS = [
#     "january", "february", "march",
#     "april", "may", "june", "july",
#     "august", "september", "october",
#     "november", "december"
# ]
# ABBR_MONTHS = [
#     "jan", "feb", "mar",
#     "apr", "may", "jun", "jul",
#     "aug", "sep", "oct",
#     "nov", "dec"
# ]
# STOP_WORDS = set(stopwords.words("english"))

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
        "word.istitle":  word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.is_year": (
            word.isdigit() and int(word) >= 1900 and int(word) >= 2100
        ),
        "word.len": len(word),
        "word.is_abbr_month": (word_lower in ABBR_MONTHS),
        "word.is_full_month": (word_lower in FULL_MONTHS),
        "word.syllable_count": syllable_count(word_lower),
        "word.first_three": word_lower[:3],
        "word.last_three": word_lower[-3:],
        "word.has_at": ("@" in word)
    }
    if idx > 0:
        prev_word = str(doc[idx-1])
        prev_word_lower = prev_word.lower()
        features.update({
            "-1:word.istitle": prev_word.istitle(),
            "-1:word.is_year": (
                prev_word.isdigit() and int(prev_word) >= 1900
                and int(prev_word) >= 2100
            ),
            "-1:word.is_abbr_month": (prev_word_lower in ABBR_MONTHS),
            "-1:word.is_full_month": (prev_word_lower in FULL_MONTHS),
            "-1:word.isdigit": prev_word.isdigit(),
            "-1:word.syllable_count": syllable_count(prev_word_lower),
            "-1:word.first_three": prev_word_lower[:3],
            "-1:word.last_three": prev_word_lower[-3:],
            "-1:word.lower": prev_word_lower,
            "-1:word.bigram": prev_word_lower + " " + word_lower
        })
    if idx < len(doc) - 1:
        next_word = str(doc[idx+1])
        next_word_lower = next_word.lower()
        features.update({
            "+1:word.istitle": next_word.istitle(),
            "+1:word.is_year": (
                next_word.isdigit() and int(next_word) >= 1900
                and int(next_word) <= 2100
            ),
            "+1:word.is_abbr_month": (next_word_lower in ABBR_MONTHS),
            "+1:word.is_full_month": (next_word_lower in FULL_MONTHS),
            "+1:word.isdigit": next_word.isdigit(),
            "+1:word.syllable_count": syllable_count(next_word_lower),
            "+1:word.first_three": next_word_lower[:3],
            "+1:word.last_three": next_word_lower[-3:],
            "+1:word.lower": next_word_lower,
            "+1:word.bigram": word_lower + " " + next_word_lower
        })
    return features


if __name__ == "__main__":
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
        train_test_split(doc_ids, [0]*len(doc_ids), test_size=.25)
    x_train = docs_to_features(x_raw_train_ids, df)
    y_train = docs_to_labels(x_raw_train_ids, df)
    x_test = docs_to_features(x_raw_test_ids, df)
    y_test = docs_to_labels(x_raw_test_ids, df)

    flat_y_test = []
    for y_array in y_test:
        for y in y_array:
            flat_y_test.append(y)

    labels = np.unique(flat_y_test)

    crf = CRF(
        algorithm="lbfgs",
        max_iterations=100,
        all_possible_transitions=True,
        all_possible_states=True,
    )
    params_space = {
        "c1": scipy.stats.expon(scale=0.5),
        "c2": scipy.stats.expon(scale=0.05)
    }
    f1_scorer = make_scorer(
        metrics.flat_f1_score,
        average="weighted",
        labels=labels
    )
    rs = RandomizedSearchCV(
        crf, params_space,
        cv=5, verbose=5,
        n_jobs=-1, n_iter=100,
        scoring=f1_scorer
    )
    train_start = process_time()
    rs.fit(x_train, y_train)
    print("Best parameters:",rs.best_params_)
    print("Best CV score:",rs.best_score_)
    crf = rs.best_estimator_
    # crf.fit(x_train, y_train)
    joblib.dump(crf, "model.pkl", compress=9)

    # Reports
    y_pred = crf.predict(x_test)
    print(sorted(crf.classes_))
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
