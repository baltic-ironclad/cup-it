import os

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_train = pd.read_csv(os.path.join('data', 'train_n.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join('data', 'test_n.csv'), encoding='utf-8')

cv = CountVectorizer()
analyzer = cv.build_analyzer()

stop_words = ['не']

sentences = df_train['text'].values
y = df_train['type'].values
classes = list(set(y))


testing = False
# Тут разбиваю тренировочные данные в соотношении 9:1, это необходимо, чтобы проверять работоспособность сети.
# Обычно на данных, на которых мы обучаем, при тестировании выходит намного более высокая точность, нежели на новых.
if testing:
    sentences_train, sentences_score, y_train, y_score = train_test_split(sentences, y, test_size=0.1)
else:
    sentences_train, y_train = sentences, y
sentences_test = df_test['text'].values

vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 1))

clf = Pipeline([
('vect', vectorizer),
# ('tfidf', TfidfTransformer()),
('clf', SGDClassifier()),
])

clf.fit(sentences_train, y_train)

print(len(vectorizer.vocabulary_))

acc = clf.score(sentences_train, y_train)
print('Accuracy for train data: {:.4f}'.format(acc))

if testing:
    predicted = clf.predict(sentences_score)
    acc = np.mean(predicted == y_score)
    print('Accuracy for score data: {:.4f}'.format(acc))

# print(metrics.classification_report(y_score, predicted, target_names=classes))
answer = clf.predict(sentences_test)
df_answer = pd.DataFrame({'type':  answer})
df_answer.to_csv('answer.csv', index=True)
