import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class CountCorrCalculator:
    def top(self, series_clean: pd.Series, n: int=10) -> pd.Series:
        cv = CountVectorizer(min_df=5)
        X  = cv.fit_transform(series_clean)
        tf = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out())
        corr = tf.corr().stack().sort_values(ascending=False).drop_duplicates()
        # excluye auto-correlaciones (=1.0)
        return corr[corr < 1.0].head(n)
