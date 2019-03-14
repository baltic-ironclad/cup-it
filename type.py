import os

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_train = pd.read_csv(os.path.join('data', 'train_n.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join('data', 'test_data.csv'), encoding='utf-8')

cv = CountVectorizer()
analyzer = cv.build_analyzer()

sentences = df_train['text'].values
y = df_train['type'].values
classes = list(set(y))

# Тут разбиваю тренировочные данные в соотношении 9:1, это необходимо, чтобы проверять работоспособность сети.
# Обычно на данных, на которых мы обучаем, при тестировании выходит намного более высокая точность, нежели на новых.
sentences_train, sentences_score, y_train, y_score = train_test_split(sentences, y, test_size=0.1)
sentences_test = df_test['text'].values

clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          random_state=42,
                          max_iter=100)),
])


clf.fit(sentences_train, y_train)

train_score = clf.score(sentences_train, y_train)
print('Accuracy for train data: {:.4f}'.format(train_score))

predicted = clf.predict(sentences_score)
print('Accuracy for score data: {:.4f}'.format(np.mean(predicted == y_score)))

print(metrics.classification_report(y_score, predicted, target_names=classes))
answer = clf.predict(sentences_test)
df_answer = pd.DataFrame({'type':  answer})
df_answer.to_csv('answer.csv', index=True)