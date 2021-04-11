from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pre_process.abstract_pre_processor import AbstractPreProcessor
from data_type.sentence import Sentence, ProcessSentence
from pre_process.pos_tag_pre_processor import PosTagPreProcessor

class LemmaPreProcessor(AbstractPreProcessor):
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def transform(self, sent: Sentence, process_sent: ProcessSentence = None) -> Sentence:
        if process_sent is None or process_sent.type!="POS":
            process_sent = PosTagPreProcessor().transform(sent)
        lemmer = WordNetLemmatizer()
        lemmas = []
        for word, pos in process_sent:
            pos = self.get_wordnet_pos(pos)
            if pos == '':
                lemmas.append(word)
            else:
                lemmas.append(lemmer.lemmatize(word=word, pos=pos))
        return ProcessSentence(lemmas, "LEMMA")

