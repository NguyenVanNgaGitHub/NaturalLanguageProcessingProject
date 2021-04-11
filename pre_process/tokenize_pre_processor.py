from nltk import word_tokenize
from pre_process.abstract_pre_processor import AbstractPreProcessor
from data_type.sentence import Sentence,ProcessSentence

class WordTokenizePreProcessor(AbstractPreProcessor):
    def transform(self, sent: Sentence, process_sent: ProcessSentence = None) -> Sentence:
        return ProcessSentence(word_tokenize(sent), "TOKEN")

