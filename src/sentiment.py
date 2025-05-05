from sentiment_analysis_spanish import sentiment_analysis

class LexiconSentiment:
    def __init__(self):
        self.analyzer      = sentiment_analysis.SentimentAnalysisSpanish()
        self.neg_words     = {"no","nunca","jamás","sin"}
        self.intensifiers  = {
          "enhorabuena","feliz","excelente","maravilla",
          "genial","gracias","gran"
        }

    def polarity(self, clean_text: str) -> float:
        p = self.analyzer.sentiment(clean_text)
        toks = clean_text.split()
        # invierte si hay negación
        if any(w in self.neg_words for w in toks):
            p = 1 - p
        # amplifica si hay intensificador
        if any(w in self.intensifiers for w in toks):
            p = min(p * 1.2, 1.0)
        return p

    def label(self, p: float) -> str:
        if p > 0.6:
            return "pos"
        elif p < 0.4:
            return "neg"
        else:
            return "neu"
