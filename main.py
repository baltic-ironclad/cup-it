import os

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv(os.path.join('data', 'train_data.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join('data', 'test_data.csv'), encoding='utf-8')


sentences_train = df_train['text'].values + df_train['title'].values
sentences_test = df_test['text'].values + df_test['title'].values
y_train = df_train['type'].values

# sentences_train, sentences_test, y_train, y_test = train_test_split(
#         sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print('Accuracy for {} data: {:.4f}'.format('source', score))
print(X_train)

answer = classifier.predict(X_test)

print(len(answer), type(answer), answer)

df_answer = pd.DataFrame({'type':  answer})

df_answer.to_csv('lol.csv')
