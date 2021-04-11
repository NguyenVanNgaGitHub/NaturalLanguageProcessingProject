from pre_process.tokenize_pre_processor import WordTokenizePreProcessor
from pre_process.pos_tag_pre_processor import PosTagPreProcessor
from pre_process.stem_pre_processor import StemPreProcessor
from pre_process.lemma_pre_process import LemmaPreProcessor

sent = "She does not like you. Doing well"
token = WordTokenizePreProcessor().transform(sent)
pos = PosTagPreProcessor().transform(sent,token)
stem = StemPreProcessor().transform(sent,token)
lemma = LemmaPreProcessor().transform(sent,pos)
print(token)
print(pos)
print(stem)
print(lemma)
