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

def get_word_labels(doc, doc_df):
    data = []
    # words = re.finditer(r"[^\s.,;:?/!)(•-]+", doc)
    words = re.finditer(r"[^\s.➢\",.:;)(\-\–•“”*']+", doc)
    for _word in words:
        pos_start = _word.start()
        word = _word.group()
        matches = doc_df[
            (doc_df["start"] <= pos_start)
            & (doc_df["end"] >= pos_start)
        ]
        label = matches['label'].iloc[0]\
            if matches.shape[0] > 0 and matches['label'].iloc[0] != "JOB_DESCRIPTION"\
            else None
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

    #data_path = input("Enter the NER data path (JSON): ")
    data_path = "../spanish_annotated_docs_v2/admin.jsonl"
    with open(data_path, "r") as data_file:
        df = pd.DataFrame({})
        while (len(raw_record := data_file.readline()) > 0):
            record = json.loads(raw_record)
            doc_id = record["id"]
            for annotation in record["label"]["entities"]:
                df = pd.concat([df,
                    pd.DataFrame([{
                        "doc_id": doc_id,
                        "doc": record["data"],
                        "start": annotation["start_offset"],
                        "end": annotation["end_offset"],
                        "label": annotation["label"]
                    }])
                ])
    doc_ids = df["doc_id"].unique()

    start = process_time()
    logging.info(f"Annotation Frame: {process_time() - start}s")
    start = process_time()
    with Pool(processes=WORKERS) as pool:
        _results = [
            pool.apply_async(get_word_labels, (
                df[df["doc_id"] == doc_id]["doc"].iloc[0],
                df[df["doc_id"] == doc_id]
            )) for doc_id in doc_ids
        ]
        results = [res.get(timeout=10) for res in _results]
    logging.info(f"Label all words: {process_time() - start}s")
    with open("output.csv", "w") as output:
        csvwriter = csv.writer(output)
        csvwriter.writerow(["doc_id","token", "label"])
        for doc_id, doc in enumerate(results):
            [csvwriter.writerow([doc_id, res[0], res[1]]) for res in doc]

