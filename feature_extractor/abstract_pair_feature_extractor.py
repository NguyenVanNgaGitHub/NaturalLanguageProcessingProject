from abc import ABC, abstractmethod
from typing import List
from data_type.sentence_pair import SentencePair, ProcessSentencePair

class AbstractPairFeatureExtractor(ABC):
    @abstractmethod
    def transform(self, sentence_pair: SentencePair, process_pair: ProcessSentencePair) -> float:
        return NotImplementedError