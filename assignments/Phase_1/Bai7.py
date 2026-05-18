import math
from Bai1 import WhitespaceTokenizer


class IDF:
    def __init__(self, smooth=False):
        self.smooth = smooth
        self.tokenizer = WhitespaceTokenizer()

    def compute(self, token, documents):
        token = token.lower()
        tokens_list = [self.tokenizer.tokenize(document) for document in documents]
        df = sum(1 for tokens in tokens_list if token in tokens)
        N = len(documents)

        if self.smooth:
            return math.log((N + 1) / (df + 1)) + 1
        return math.log(N / df) if df > 0 else 0.0


# idf = IDF()
# documents = ["NLP is fun", "I love NLP", "NLP NLP NLP"]
# print(idf.compute(token="nlp", documents=documents))
