from nltk.stem import PorterStemmer
from pre_process.abstract_pre_processor import AbstractPreProcessor
from data_type.sentence import Sentence, ProcessSentence
from pre_process.tokenize_pre_processor import WordTokenizePreProcessor

class StemPreProcessor(AbstractPreProcessor):
    def transform(self, sent: Sentence, process_sent: ProcessSentence = None) -> Sentence:
        if process_sent is None or process_sent.type!="TOKEN":
            process_sent = WordTokenizePreProcessor().transform(sent)
        stemmer = PorterStemmer()
        return ProcessSentence([stemmer.stem(word) for word in process_sent], "STEM")