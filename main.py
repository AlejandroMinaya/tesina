"""
Tesina
Juan Alejandro Alcántara Minaya
Feb - Jun 2022
"""

"""
Dataset
Resume Entities for NER
https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner

The dataset has 220 items of which 220 items have been manually labeled.
The labels are divided into following 10 categories:
- Name
- College Name
- Degree
- Graduation Year
- Years of Experience
- Companies worked at
- Designation
- Skills
- Location
- Email Address
"""

"""
Sentences can either be decided by window size (number of words) or by a
delimiter different to the space, e.g. a period.
"""
import pandas as pd
import json

def word_to_features(sentence, idx):
    word = sentence[idx]
    word_lower = word.lower()
    features = [
        "bias",
        "word.istitle=" + word.istitle(),
        f"word.university={(word_lower == 'university')}",
        f"word.college={(word_lower == 'college')}",
        f"word.isdigit={word.isdigit()}",
        f"word.len={len(word)}"
    ]
    if idx > 0:
        prev_word = sentence[idx-1]
        prev_word_lower = prev_word.lower()
        features.extend([
            "-1:word.istitle=" + prev_word.istitle(),
            f"-1:word.university={(prev_word_lower == 'university')}",
            f"-1:word.college={(prev_word_lower == 'college')}",
            f"-1:word.isdigit={prev_word.isdigit()}"
        ])

    if idx < len(sentence) - 1:
        next_word = sentence[idx+1]
        next_word_lower = next_word.lower()
        features.extend([
            "+1:word.istitle=" + next_word.istitle(),
            f"+1:word.university={(next_word_lower == 'university')}",
            f"+1:word.college={(next_word_lower == 'college')}",
            f"+1:word.isdigit={next_word.isdigit()}"
        ])
    return features

def sentence_to_features(_sentence):
    sentence = _sentence.split()
    return [
        word_to_features(sentence, idx) for idx in range(len(sentence))
    ]


def main():
    # Open the JSON file with all the records
    with open("resume_dataset.json") as dataset:
        data = json.load(dataset)
    # Flatten the information to ease the data ingestion
    for idx, record in enumerate(data):
        for annotation in record["annotation"]:
            if len(annotation["label"]) > 0:
                annotation["label"] = annotation["label"][0]
            else:
                annotation["label"] = "UNKNOWN"
            annotation["start"] = annotation["points"][0]["start"]
            annotation["end"] = annotation["points"][0]["end"]
            annotation["text"] = annotation["points"][0]["text"]
            annotation["record_id"] = idx
            del annotation["points"]
    data = pd.json_normalize(
        data,
        "annotation",
        ["content", "extras"]
    )
    # Change label into categorical data
    data["label"] = data["label"].astype("category")
    # Re-index the dataframe based on the record and the annotation position
    data.index = pd.MultiIndex.from_tuples(
        zip(data["record_id"], data["start"], data["end"]),
        names=["id", "start", "end"]
    )
    # Keep only the columns we are interested in
    annotations = data[["label", "text", "content"]]
    print(data.info())

if __name__ == "__main__":
    main()
