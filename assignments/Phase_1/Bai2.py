import re
from underthesea import word_tokenize, sent_tokenize
from underthesea.datasets import stopwords


class VietnameseTextProcessor:
    def sentence_tokenize(self, text):
        return sent_tokenize(text)

    def word_tokenize(self, text):
        return word_tokenize(text)

    def remove_urls(self, text):
        return re.sub(r"http[s]?://\S+|www\.\S+", "", text)

    def remove_html(self, text):
        return re.sub(r"<[^>]+>", "", text)

    def remove_emojis(self, text):
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    def remove_punctuation(self, text):
        return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    def normalize_whitespace(self, text):
        return re.sub(r"\s+", " ", text)

    def remove_stopwords(self, text):
        tokens = word_tokenize(text)

        return [token for token in tokens if token.lower() not in stopwords.words]

    def preprocess(self, text):
        text = text.lower()
        text = self.remove_urls(text)
        text = self.remove_html(text)
        text = self.remove_emojis(text)
        text = self.remove_punctuation(text)
        text = self.normalize_whitespace(text)

        tokens = self.remove_stopwords(text)

        return tokens


text = """
<p>Xin chào!!!</p>
Tôi đang học NLP 😄
Hãy ghé thăm https://abc.com ngay bây giờ!!!
"""
nlp = VietnameseTextProcessor()
print(nlp.preprocess(text))
