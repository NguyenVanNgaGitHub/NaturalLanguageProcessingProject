from feature_extractor.abstract_pair_feature_extractor import AbstractPairFeatureExtractor
from data_type.sentence_pair import SentencePair, ProcessSentencePair

class SimpleMatchingExtractor(AbstractPairFeatureExtractor):

    def transform(self, sentence_pair: SentencePair, process_pair: ProcessSentencePair) -> float:
        if not hasattr(process_pair.text, "set"):
            process_pair.text.create_set()
        if not hasattr(process_pair.hypothesis, "set"):
            process_pair.hypothesis.create_set()
        return len(process_pair.hypothesis.set.intersection(process_pair.text.set))/len(process_pair.hypothesis.set)

