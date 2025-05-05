import re
import spacy
import nltk
from nltk.corpus import stopwords

# Descarga (una sola vez)
nltk.download("stopwords", quiet=True)

# Carga modelo spaCy y stopwords
nlp     = spacy.load("es_core_news_sm")
stop_es = set(stopwords.words("spanish"))

class SpacyPreprocessor:
    def clean(self, text: str) -> str:
        # minusculas, quita URLs, menciones y símbolos no alfabéticos
        txt = re.sub(r"http\S+|@\w+|[^A-Za-zÁÉÍÓÚáéíóúÑñ\s]", " ", text.lower())
        doc = nlp(txt)
        tokens = [
            tok.lemma_
            for tok in doc
            if tok.lemma_ not in stop_es
               and not tok.is_punct
               and len(tok) > 1
        ]
        return " ".join(tokens)
