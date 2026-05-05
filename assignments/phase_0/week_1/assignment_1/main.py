from collections import Counter
from typing import Iterable


def compute_word_frequency(documents: Iterable[str]) -> dict[str, int]:
    words = " ".join(documents).split()
    return dict(Counter(words))


if __name__ == "__main__":
    documents = [
        "the quick brown fox",
        "the lazy dog sleeps",
        "the fox jumps over the dog",
    ]
    print(compute_word_frequency(documents))
