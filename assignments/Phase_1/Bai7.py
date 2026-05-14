import math

from Bai1 import WhitespaceTokenizer


class IDF:
    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()

    def compute(self, token, documents):
        token = token.lower()
        tokens_list = [self.tokenizer.tokenize(document) for document in documents]
        df = sum(1 for tokens in tokens_list if token in tokens)
        return math.log(len(documents) / df)


idf = IDF()
documents = ["NLP is fun", "I love NLP", "NLP NLP NLP"]
print(idf.compute(token="nlp", documents=documents))
