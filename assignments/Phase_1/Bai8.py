import math
import json
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
        vector = [0.0] * len(self.vocab)

        for token, tf_val in tf_scores.items():
            if token in self.vocab:
                idf_val = self.idf.compute(token, documents)
                vector[self.vocab[token]] = tf_val * idf_val

        if self.normalize:
            norm = math.sqrt(sum(v**2 for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]

        return vector

    def fit_transform(self, documents):
        matrix = [self.compute_vector(doc, documents) for doc in documents]
        return matrix, self.vocab

    def print_matrix(self, matrix):
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

    model = TFIDF(vocab_path="vocab.json")
    matrix, vocab = model.fit_transform(documents)
    model.print_matrix(matrix)
    print(f"\nVocabulary: {vocab}")

    print("\n=== Smooth + Normalized TF-IDF ===")
    model2 = TFIDF(vocab_path="vocab.json", smooth=True, normalize=True)
    matrix2, _ = model2.fit_transform(documents)
    model2.print_matrix(matrix2)
