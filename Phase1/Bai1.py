import re


class WhitespaceTokenizer:

    def lowercase(self, text):
        return text.lower()

    def normalize_whitespace(self, text):
        return re.sub(r'\s+', ' ', text)

    def remove_punctuation(self, text):
        return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    def tokenize(self, text):
        text = self.lowercase(text)
        text = self.normalize_whitespace(text)
        text = self.remove_punctuation(text)
        return text.split()

    def batch_tokenize(self, texts):
        return [self.tokenize(text) for text in texts]

tokenizer = WhitespaceTokenizer()

text = '   Hello    word   '
print(tokenizer.tokenize(text))

batch_text = ['   Hello   word  ', 'I am learning NLP']
print(tokenizer.batch_tokenize(batch_text))