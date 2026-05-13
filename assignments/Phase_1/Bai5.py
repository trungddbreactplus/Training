import json
import numpy as np

from Bai1 import WhitespaceTokenizer


class BagOfWords:
    def __init__(self, vocab_path: str, mode="count"):

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.vocab_size = len(self.vocab)

        self.mode = mode

        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]

        self.tokenizer = WhitespaceTokenizer()

    def encode(self, text: str):

        tokens = self.tokenizer.tokenize(text)

        vector = np.zeros(self.vocab_size, dtype=np.int32)

        for token in tokens:

            token_id = self.vocab.get(token, self.unk_id)

            if token_id == self.pad_id:
                continue

            if self.mode == "binary":
                vector[token_id] = 1

            else:
                vector[token_id] += 1

        return vector


bow = BagOfWords(
    vocab_path="vocab.json",
    mode="count"
)

documents = [
    "NLP is fun",
    "I love NLP",
    "NLP NLP NLP"
]
for document in documents:
    print(bow.encode(document))