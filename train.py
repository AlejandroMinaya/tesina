#!/usr/bin/env python
"""
Training the CRF model
Tesina
Juan Alejandro Alcántara Minaya
Feb - Jun 2022
"""
from sklearn.model_selection import train_test_split
from sys import stdout
import csv
import logging as _logging
import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random


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
        "word.len": len(word),
        "word.first_letter": word[0],
        "word.last_letter": word[:-1],
        "word.syllable_count": syllable_count(word_lower)
    }
    if idx > 0:
        prev_word = str(doc[idx-1])
        prev_word_lower = prev_word.lower()
        features.update({
            "-1:word.istitle": prev_word.istitle(),
            "-1:word.isdigit": prev_word.isdigit(),
            "-1:word.first_letter": prev_word[0],
            "-1:word.last_letter": prev_word[-1],
            "-1:word.syllable_count": syllable_count(prev_word_lower)
        })
    if idx < len(doc) - 1:
        next_word = str(doc[idx+1])
        next_word_lower = next_word.lower()
        features.update({
            "+1:word.istitle": next_word.istitle(),
            "+1:word.isdigit": next_word.isdigit(),
            "+1:word.first_letter": next_word[0],
            "+1:word.last_letter": next_word[-1],
            "+1:word.syllable_count": syllable_count(next_word_lower)
        })
    return features


if __name__ == "__main__":
    # Logger set-up
    logging = _logging.getLogger()
    logging.setLevel(0)
    handler = _logging.StreamHandler(stdout)
    handler.setLevel(0)
    handler.setFormatter(_logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s"
    ))
    logging.addHandler(handler)

    data_path = input("Enter the data path (CSV): ")
    df = pd.read_csv(data_path, header=0)
    doc_ids = list(df["doc_id"].unique())
    random.shuffle(doc_ids)

    x_raw_train_ids, x_raw_test_ids, _, _ =\
        train_test_split(doc_ids, [0]*len(doc_ids), test_size=.33)
    x_train = docs_to_features(x_raw_train_ids, df)
    y_train = docs_to_labels(x_raw_train_ids, df)
    x_test = docs_to_features(x_raw_test_ids, df)
    y_test = docs_to_labels(x_raw_test_ids, df)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)
    labels = list(crf.classes_)
    y_pred = crf.predict(x_test)
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels))

