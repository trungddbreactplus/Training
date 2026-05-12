import json
import numpy as np

from Bai1 import WhitespaceTokenizer


class Embedding_OneHot:

    def __init__(self, vocab_path: str):

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.vocab_size = len(self.vocab)

        self.pad_idx = self.vocab["<PAD>"]
        self.unk_idx = self.vocab["<UNK>"]

        self.tokenizer = WhitespaceTokenizer()

    def onehot(
        self,
        batch_text,
        max_sequence_length=None,
        padding=True,
        truncation=True
    ):

        batch_vectors = []

        for text in batch_text:

            tokens = self.tokenizer.tokenize(text)

            if truncation and max_sequence_length:
                tokens = tokens[:max_sequence_length]

            sentence_vectors = []

            for token in tokens:

                idx = self.vocab.get(token, self.unk_idx)

                vector = np.zeros(self.vocab_size, dtype=int)

                vector[idx] = 1

                sentence_vectors.append(vector)

            if padding and max_sequence_length:

                pad_vector = np.zeros(self.vocab_size, dtype=int)
                pad_vector[self.pad_idx] = 1

                while len(sentence_vectors) < max_sequence_length:
                    sentence_vectors.append(pad_vector)

            batch_vectors.append(np.array(sentence_vectors))

        return np.array(batch_vectors)


onehot = Embedding_OneHot(vocab_path="./vocab.json")

sentences = [
    "I love NLP",
    "NLP is very fun"
]

vectors = onehot.onehot(
    batch_text=sentences,
    max_sequence_length=6,
    padding=True,
    truncation=True
)

print(vectors.shape)

for v in vectors:
    print(v)
    print()