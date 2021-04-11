from abc import ABC, abstractmethod
from data_type.sentence import Sentence, ProcessSentence

class AbstractPreProcessor(ABC):
    @abstractmethod
    def transform(self, sent: Sentence, process_sent: ProcessSentence = None) -> Sentence:
        return NotImplementedError