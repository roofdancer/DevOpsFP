import nltk
import numpy as np
import string
import pymorphy2


class SimpleClassifier:
    def __init__(self):
        self.word_tokenizer = nltk.WordPunctTokenizer()
        self.dates = [str(x) for x in np.arange(1900, 2050)]
        self.morph = pymorphy2.MorphAnalyzer()

    def classify(self, sent):
        text_lower = sent.lower()
        tokens = self.word_tokenizer.tokenize(text_lower)
        tokens = [word for word in tokens if
                  (word not in string.punctuation and word not in self.dates)]
        for token in tokens:
            lemma = self.morph.parse(token)
            for option in lemma:
                for method in option.methods_stack:
                    if type(method[0]).__name__ == 'FakeDictionary':
                        return 0
        return 1
