import json
import numpy as np
from Bai6 import TF
from Bai7 import IDF


class TFIDF:
    def __init__(self, vocab_path, tf_mode="norm", smooth=False, normalize=False):
        self.tf = TF(mode=tf_mode)
        self.idf = IDF(smooth=smooth)
        self.normalize = normalize
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def compute_vector(self, document, documents):
        tf_scores = self.tf.compute(document)
        vector = np.zeros(len(self.vocab))

        for token, tf_val in tf_scores.items():
            if token in self.vocab:
                idf_val = self.idf.compute(token, documents)
                vector[self.vocab[token]] = tf_val * idf_val

        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def fit_transform(self, documents):
        matrix = np.array([self.compute_vector(doc, documents) for doc in documents])
        return matrix, self.vocab

    def print_matrix(self, matrix, documents):
        terms = sorted(self.vocab, key=self.vocab.get)
        print(f"{'':25}" + "".join(f"{t:>10}" for t in terms))
        print("-" * (25 + 10 * len(terms)))
        for doc, row in zip(documents, matrix):
            print(f'"{doc}"'[:24].ljust(25) + "".join(f"{v:>10.4f}" for v in row))


if __name__ == "__main__":
    documents = [
        "I love NLP",
        "NLP is fun",
        "I love machine learning",
    ]

    model = TFIDF(vocab_path="vocab.json", smooth=True, normalize=True)
    matrix, vocab = model.fit_transform(documents)

    print(f"vocab: {vocab}")
    print(f"matrix shape: {matrix.shape}\n")
    model.print_matrix(matrix, documents)
