from Bai1 import WhitespaceTokenizer
from collections import Counter


class TF:
    def __init__(self, mode):
        self.mode = mode
        self.tokenizer = WhitespaceTokenizer()

    def compute(self, document):
        tokens = self.tokenizer.tokenize(document)
        token_counts = Counter(tokens)

        if self.mode == "count":
            return dict(token_counts)

        elif self.mode == "norm":
            total_tokens = len(tokens)
            return {
                token: count / total_tokens for token, count in token_counts.items()
            }
        return None


# tf_norm = TF(mode="norm")
# print(tf_norm.compute("NLP NLP is fun"))
#
# tf_count = TF(mode="count")
# print(tf_count.compute("NLP NLP is fun"))
