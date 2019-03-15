import pandas as pd

import os

from sklearn.feature_extraction.text import CountVectorizer

def get_word_frequencies():
    df = pd.read_csv(os.path.join('data', 'train_n.csv'), encoding='utf-8')
    texts = df['text'].values
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(texts)
    freq = cv_fit.toarray().sum(axis=0)
    word_freq = sorted(list(zip(freq, cv.get_feature_names())))

    print(len(word_freq))

    print(len([word for word in word_freq if word[0] == 1]))


if __name__ == '__main__':
    get_word_frequencies()
