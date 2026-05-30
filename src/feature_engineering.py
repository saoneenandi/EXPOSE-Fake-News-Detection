from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf():

    return TfidfVectorizer(
        max_features=10000,
        stop_words="english"
    )
