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
            "\U0001F600-\U0001F64F" 
            "\U0001F300-\U0001F5FF"  
            "\U0001F680-\U0001F6FF"  
            "\U0001F1E0-\U0001F1FF"  
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_punctuation(self, text):
        return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    def normalize_whitespace(self, text):
        return re.sub(r'\s+', ' ', text)

    def remove_stopwords(self, text):
        tokens = word_tokenize(text)

        return [
            token for token in tokens
            if token.lower() not in stopwords.words
        ]

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
