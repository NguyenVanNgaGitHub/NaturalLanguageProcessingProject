from pre_process.tokenize_pre_processor import WordTokenizePreProcessor
from pre_process.pos_tag_pre_processor import PosTagPreProcessor
from pre_process.stem_pre_processor import StemPreProcessor
from pre_process.lemma_pre_process import LemmaPreProcessor
from feature_extractor.lexical_feature.lavenshtein_distance_extractor import LavenshteinExtractor
from feature_extractor.lexical_feature.simple_matching_extractor import SimpleMatchingExtractor
from feature_extractor.lexical_feature.tri_gram_character_extractor import TriGramCharacterExtractor
from feature_extractor.lexical_feature.rouge_s_extractor import RougeSExtractor
from feature_extractor.lexical_feature.consecutive_subsequence_matching_extractor import ConsecutiveSubsequenceMatchingExtractor
from data_type.sentence_pair import SentencePair, ProcessSentencePair

sent1 = "She does not like you. Poor"
token1 = WordTokenizePreProcessor().transform(sent1)
pos1 = PosTagPreProcessor().transform(sent1,token1)
stem1 = StemPreProcessor().transform(sent1,token1)
lemma1 = LemmaPreProcessor().transform(sent1,pos1)

sent2 = "She like you. Doing well"
token2 = WordTokenizePreProcessor().transform(sent2)
pos2 = PosTagPreProcessor().transform(sent2,token2)
stem2 = StemPreProcessor().transform(sent2,token2)
lemma2 = LemmaPreProcessor().transform(sent2,pos2)

sent_pair = SentencePair(sent1, sent2)
sent_process_pair = ProcessSentencePair(token1,token2)
print(SimpleMatchingExtractor().transform(sent_pair, sent_process_pair))
print(LavenshteinExtractor().transform(sent_pair, sent_process_pair))
print(TriGramCharacterExtractor().transform(sent_pair, sent_process_pair))
print(RougeSExtractor().transform(sent_pair, sent_process_pair))
print(ConsecutiveSubsequenceMatchingExtractor().transform(sent_pair, sent_process_pair))



