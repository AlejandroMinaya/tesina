#!/usr/bin/env python
"""
Query the CRF model
Tesina
Juan Alejandro Alcántara Minaya
Feb - Jun 2022
"""
import joblib
import re
from train import word_to_features

if __name__ == "__main__":
    # model_path = input("Enter the model path (PKL): ")
    model_path = "model.pkl"
    # input_path = input("Enter input path (TXT): ")
    input_path = "sample_input.txt"
    crf = joblib.load(model_path)
    with open(input_path) as f:
        words = [
            match.group()\
            for match in re.finditer("\S+",f.read())
        ]
    x = [
        word_to_features(words, i)\
        for i in range(len(words))
    ]

    preds = crf.predict([x])[0]
    for w, p in zip(words, preds):
        if 'nan' not in p:
            print(f"{w} -> {p}")

