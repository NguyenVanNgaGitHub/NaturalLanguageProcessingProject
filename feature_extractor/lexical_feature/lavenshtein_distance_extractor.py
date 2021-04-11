from feature_extractor.abstract_pair_feature_extractor import AbstractPairFeatureExtractor
from data_type.sentence_pair import SentencePair, ProcessSentencePair
from nltk.metrics import edit_distance

class LavenshteinExtractor(AbstractPairFeatureExtractor):

    def transform(self, sentence_pair: SentencePair, process_pair: ProcessSentencePair) -> float:
        if not hasattr(process_pair.text, "set"):
            process_pair.text.create_set()
        if not hasattr(process_pair.hypothesis, "set"):
            process_pair.hypothesis.create_set()
        sum = 0
        for w_h in process_pair.hypothesis.set:
            match = 0
            for w_t in process_pair.text.set:
                distance = edit_distance(w_h, w_t)
                if distance == 0:
                    match = 1
                    break
                elif distance == 1:
                    match = 0.9
                else:
                    match = max(match, 1/distance)
            sum+=match
        return sum/len(process_pair.hypothesis.set)


