import pandas as pd

import os

from sklearn.feature_extraction.text import CountVectorizer

def get_n_most_frequent_words(filePath, columns, n):
    df = pd.read_csv(filePath, encoding='utf-8')
    texts = ''
    for column in columns:
        texts += ' ' + df[column].values
    cv = CountVectorizer()
    texts_fit = cv.fit_transform(texts)
    freq = texts_fit.toarray().sum(axis=0)
    word_freq = sorted(list(zip(freq, cv.get_feature_names())))
    word_freq.reverse()
    print(word_freq[:n])


if __name__ == '__main__':
    filePath = os.path.join('data', 'train_n.csv')
    columns = ['text']
    n = 20
    get_n_most_frequent_words(filePath, columns, n)
