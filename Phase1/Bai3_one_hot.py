import json
import numpy as np

from Bai1 import WhitespaceTokenizer


class Embedding_OneHot:
    def __init__(self, vocab_path:str):

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.vocab_size = len(self.vocab)

        self.unk_idx = self.vocab["<UNK>"]

        self.tokenizer = WhitespaceTokenizer()

    def onehot(self, batch_text):

        batch_vectors = []

        for text in batch_text:

            tokens = self.tokenizer.tokenize(text)

            sentence_vectors = []

            for token in tokens:

                idx = self.vocab.get(token, self.unk_idx)

                vector = np.zeros(self.vocab_size)

                vector[idx] = 1

                sentence_vectors.append(vector)

            batch_vectors.append(np.array(sentence_vectors))

        return batch_vectors


onehot = Embedding_OneHot(vocab_path="./vocab.json")

sentences = [
    "I love NLP",
    "NLP is very fun"
]

vectors = onehot.onehot(sentences)

for v in vectors:
    print(v)
    print()