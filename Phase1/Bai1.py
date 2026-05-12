import re


class WhitespaceTokenizer:

    def lowercase(self, text):
        return text.lower()

    def normalize_whitespace(self, text):
        return re.sub(r'\s+', ' ', text)

    def tokenize(self, text):
        text = self.lowercase(text)
        text = self.normalize_whitespace(text)
        return text.split()
tokenizer = WhitespaceTokenizer()
text = 'Hello    word'
print(tokenizer.tokenize(text))