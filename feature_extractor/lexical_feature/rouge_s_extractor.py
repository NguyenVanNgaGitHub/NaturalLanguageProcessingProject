from feature_extractor.abstract_pair_feature_extractor import AbstractPairFeatureExtractor
from data_type.sentence_pair import SentencePair, ProcessSentencePair
from nltk.util import skipgrams

class RougeSExtractor(AbstractPairFeatureExtractor):

    def transform(self, sentence_pair: SentencePair, process_pair: ProcessSentencePair) -> float:
        text_set = set(skipgrams(process_pair.text,2,2))
        hypothesis_set = set(skipgrams(process_pair.hypothesis,2,2))
        return len(hypothesis_set.intersection(text_set))/len(hypothesis_set)


