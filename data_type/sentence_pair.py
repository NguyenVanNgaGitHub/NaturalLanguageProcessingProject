from data_type.sentence import Sentence, ProcessSentence

class SentencePair:
    def __init__(self, text: Sentence, hypothesis: Sentence, label=0):
        self.text = text
        self.hypothesis = hypothesis
        self.label = label

class ProcessSentencePair:
    def __init__(self, text: ProcessSentence, hypothesis: ProcessSentence):
        self.text = text
        self.hypothesis = hypothesis