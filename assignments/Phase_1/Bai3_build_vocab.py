import json
from Bai1 import WhitespaceTokenizer


class BuildVocab:
    def build(self, file_path, batch_text):
        tokenizer = WhitespaceTokenizer()

        text = " ".join(batch_text)

        tokens = tokenizer.tokenize(text)

        unique_tokens = sorted(set(tokens))

        vocab = {"<PAD>": 0, "<UNK>": 1}

        for idx, token in enumerate(unique_tokens, start=2):
            vocab[token] = idx

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)


sentences = ["I love NLP", "NLP is fun"]

builder = BuildVocab()

builder.build(file_path="vocab.json", batch_text=sentences)
