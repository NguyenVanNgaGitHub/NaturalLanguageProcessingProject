from feature_extractor.abstract_pair_feature_extractor import AbstractPairFeatureExtractor
from data_type.sentence_pair import SentencePair, ProcessSentencePair

class ConsecutiveSubsequenceMatchingExtractor(AbstractPairFeatureExtractor):

    def transform(self, sentence_pair: SentencePair, process_pair: ProcessSentencePair) -> float:
        pre_matchs = []
        for i in range(0, len(process_pair.hypothesis)-1):
            for j in range(0, len(process_pair.text)-1):
                if process_pair.hypothesis[i] == process_pair.text[j]:
                    pre_matchs.append((i,j))
        fcsmatch = 0
        for i in range(1, len(process_pair.hypothesis)):
            count = 0
            this_matchs = []
            for pre_match in pre_matchs:
                if process_pair.hypothesis[pre_match[0]+i] == process_pair.text[pre_match[1]+i]:
                    if pre_match[0]+i!=len(process_pair.hypothesis)-1 and pre_match[1]+i!=len(process_pair.text)-1:
                        this_matchs.append(pre_match)
            pre_matchs = this_matchs
            fcsmatch += count/(len(process_pair.hypothesis)-i)
        return fcsmatch/(len(process_pair.hypothesis)-1)


