from typing import List

class Sentence(str):
    def __init__(self, content):
        self = content

class ProcessSentence(List):
    def __init__(self, list_element, type):
        self.extend(list_element)
        self.type = type

    def create_set(self):
        self.set = set(self)