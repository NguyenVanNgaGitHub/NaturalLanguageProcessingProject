from feature_extractor.abstract_pair_feature_extractor import AbstractPairFeatureExtractor
from data_type.sentence_pair import SentencePair, ProcessSentencePair
from nltk.util import trigrams

class TriGramCharacterExtractor(AbstractPairFeatureExtractor):
    def transform(self, sentence_pair: SentencePair, process_pair: ProcessSentencePair) -> float:
        text_set = set(trigrams(sentence_pair.text))
        hypothesis_set = set(trigrams(sentence_pair.hypothesis))
        return len(hypothesis_set.intersection(text_set))/len(hypothesis_set)


