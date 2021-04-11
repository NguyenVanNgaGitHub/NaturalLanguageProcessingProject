from sklearn.metrics import accuracy_score
from data_type.dataset import DataSet
from model.svm_model import SVMModel
from config import *
from feature_extractor.lexical_feature.consecutive_subsequence_matching_extractor import ConsecutiveSubsequenceMatchingExtractor
from feature_extractor.lexical_feature.lavenshtein_distance_extractor import LavenshteinExtractor
from feature_extractor.lexical_feature.rouge_s_extractor import RougeSExtractor
from feature_extractor.lexical_feature.simple_matching_extractor import SimpleMatchingExtractor
from feature_extractor.lexical_feature.tri_gram_character_extractor import TriGramCharacterExtractor
from data_type.sentence import Sentence, ProcessSentence
from data_type.sentence_pair import SentencePair, ProcessSentencePair
from pre_process.tokenize_pre_processor import WordTokenizePreProcessor
from pre_process.pos_tag_pre_processor import PosTagPreProcessor
from pre_process.lemma_pre_process import LemmaPreProcessor
from pre_process.stem_pre_processor import StemPreProcessor
from feature_extractor.abstract_pair_feature_extractor import AbstractPairFeatureExtractor

class Pipeline:
    def __init__(self):
        self.set_trainset_directory()
        self.set_testset_directory()
        self.set_classification_model()
        self.set_features()
        self.extract_train_features()
        self.extract_train_labels()
        self.extract_test_features()
        self.extract_test_labels()

    def set_trainset_directory(self, directory="../data/train_pairs.csv"):
        self.trainset = DataSet(directory=directory)
        self.trainset.read_data()

    def set_testset_directory(self, directory="../data/test_pairs.csv"):
        self.testset = DataSet(directory=directory)
        self.testset.read_data()

    def set_classification_model(self, type=SVM):
        if type==SVM:
            self.model = SVMModel()

    def set_features(self, features=[SIMPLE_MATCHING, LAVENSHTEIN_DISTANCE, ROUGE_S,
                                     CONSECUTIVE_SUBSEQUENCE_MATCHING, TRI_GRAM_CHARACTER]):
        self.feature_extractors = []
        if SIMPLE_MATCHING in features:
            self.feature_extractors.append(SimpleMatchingExtractor())
        if LAVENSHTEIN_DISTANCE in features:
            self.feature_extractors.append(LavenshteinExtractor())
        if ROUGE_S in features:
            self.feature_extractors.append(RougeSExtractor())
        if CONSECUTIVE_SUBSEQUENCE_MATCHING in features:
            self.feature_extractors.append(ConsecutiveSubsequenceMatchingExtractor())
        if TRI_GRAM_CHARACTER in features:
            self.feature_extractors.append(TriGramCharacterExtractor())

    def extract_features(self, dataset: DataSet):
        X = []
        for sentence_pair in dataset.sentence_pairs:
            x = []
            token_pair = ProcessSentencePair(WordTokenizePreProcessor().transform(sentence_pair.text),
                                             WordTokenizePreProcessor().transform(sentence_pair.hypothesis))
            pos_pair = ProcessSentencePair(PosTagPreProcessor().transform(sentence_pair.text, token_pair.text),
                                           PosTagPreProcessor().transform(sentence_pair.hypothesis, token_pair.hypothesis))
            stem_pair = ProcessSentencePair(StemPreProcessor().transform(sentence_pair.text, token_pair.text),
                                            StemPreProcessor().transform(sentence_pair.hypothesis, token_pair.hypothesis))
            lemma_pair = ProcessSentencePair(LemmaPreProcessor().transform(sentence_pair.text, pos_pair.text),
                                             LemmaPreProcessor().transform(sentence_pair.hypothesis, token_pair.hypothesis))

            for feature_extractor in self.feature_extractors:
                x.append(feature_extractor.transform(sentence_pair, token_pair))
                x.append(feature_extractor.transform(sentence_pair, stem_pair))
                x.append(feature_extractor.transform(sentence_pair, lemma_pair))
            X.append(x)
        return X

    def extract_labels(self, dataset: DataSet):
        Y = []
        for sentence_pair in dataset.sentence_pairs:
            Y.append(sentence_pair.label)
        return Y

    def extract_train_features(self):
        self.train_features = self.extract_features(dataset=self.trainset)

    def extract_train_labels(self):
        self.train_labels = self.extract_labels(dataset=self.trainset)

    def extract_test_features(self):
        self.test_features = self.extract_features(dataset=self.testset)

    def extract_test_labels(self):
        self.test_labels = self.extract_labels(dataset=self.testset)

    def train_classification_model(self):
        self.model.fit(self.train_features, self.train_labels)

    def test_classification_model(self):
        self.test_predicts = self.model.transform(self.test_features)
        print(accuracy_score(self.test_labels, self.test_predicts))

    def predict(self, dataset: DataSet):
        X = self.extract_features(dataset=dataset)
        return self.model.transform(X)