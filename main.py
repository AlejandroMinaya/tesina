import json
import pprint
pp = pprint.PrettyPrinter(indent=1)

dataset = json.load(open("resume_dataset.json", "r"))
annotations = [data["annotation"] for data in dataset]
pp.pprint(dataset[:5])
