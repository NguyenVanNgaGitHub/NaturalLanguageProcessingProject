import pandas as pd
from data_type.sentence import Sentence
from data_type.sentence_pair import SentencePair

class DataSet:
    def __init__(self, directory: str):
        self.directory = directory
        self.sentence_pairs = []

    def read_data(self, type="csv"):
        if type=="csv":
            data = pd.read_csv(self.directory)
            for idx, row in data.iterrows():
                text = Sentence(row["Origin"])
                hypothesis = Sentence(row["Suspect"])
                label = int(row["Label"])
                self.sentence_pairs.append(SentencePair(text,hypothesis,label))

    def print_data(self):
        for pair in self.sentence_pairs:
            print(pair.text)
            print(pair.hypothesis)
            print(pair.label)

