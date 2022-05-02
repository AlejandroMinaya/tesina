#!/usr/bin/env python
"""
Data Ingestion & Processing
Tesina
Juan Alejandro Alcántara Minaya
Feb - Jun 2022
"""
from os import path
from sys import stdout
from time import process_time
from multiprocessing import Pool, TimeoutError
import csv
import json
import logging as _logging
import pandas as pd
import re

WORKERS = 16

def open_json(data_path):
    try:
        with open(data_path, "r") as data_file:
            return json.load(data_file)
    except FileNotFoundError:
        logging.error(f"Unable to find '{path.abspath(data_path)}'")
        return None

def get_annotations_frame(json_data):
    df = pd.DataFrame({})
    for idx, doc in enumerate(json_data):
        for annotation in doc["annotation"]:
            label = annotation["label"][0] if len(annotation["label"]) > 0 else None
            df = pd.concat([df,
                pd.DataFrame([{
                    "doc_id": idx,
                    "start": annotation["points"][0]["start"],
                    "end": annotation["points"][0]["end"],
                    "label": label
                }])
            ])
    return df


def get_word_labels(doc, doc_df):
    data = []
    words = re.finditer(r"\S+", doc)
    for _word in words:
        pos_start = _word.start()
        word = _word.group()
        matches = doc_df[
            (doc_df["start"] <= pos_start)
            & (doc_df["end"] >= pos_start)
        ]
        label = matches['label'].iloc[0] if matches.shape[0] > 0 else "nan"
        data.append((word, label))
        pos_start += len(word) + 1
    return data



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

    data_path = input("Enter the NER data path (JSON): ")
    json_data = open_json(data_path)
    # If there is no data, exit with error code
    if json_data is None:
        exit(-1)

    start = process_time()
    df = get_annotations_frame(json_data)
    logging.info(f"Annotation Frame: {process_time() - start}s")
    start = process_time()
    with Pool(processes=WORKERS) as pool:
        _results = [
            pool.apply_async(get_word_labels, (
                doc["content"], df[df["doc_id"] == doc_id]
            )) for doc_id, doc in enumerate(json_data)
        ]
        results = [res.get(timeout=10) for res in _results]
    logging.info(f"Label all words: {process_time() - start}s")
    with open("output.csv", "w") as output:
        csvwriter = csv.writer(output)
        csvwriter.writerow(["doc_id","token", "label"])
        for doc_id, doc in enumerate(results):
            [csvwriter.writerow([doc_id, res[0], res[1]]) for res in doc]

