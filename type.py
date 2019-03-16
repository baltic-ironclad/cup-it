import os

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_train = pd.read_csv(os.path.join('data', 'train_n.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join('data', 'test_n.csv'), encoding='utf-8')

cv = CountVectorizer()
analyzer = cv.build_analyzer()

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

vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 1), stop_words=stopwords.words('russian'))

"""
clf = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])
"""

tf_idf_model = vectorizer.fit(np.concatenate([df_train.text, df_test.text]))
train_tfidf_vec = tf_idf_model.transform(df_train.text)
test_tfidf_vec = tf_idf_model.transform(df_test.text)

classifier = LogisticRegression(solver='liblinear')
classifier_params = {
    'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100]
}
classifier_search = GridSearchCV(
    estimator=classifier, param_grid=classifier_params
)
classifier_fitted = classifier_search.fit(X=sentences_train, y=y_train)


accuracy = classifier_fitted.score(X=sentences_train, y=y_train)
print('Accuracy for train data: {:.4f}'.format(accuracy))

if testing:
    predicted = classifier_fitted.predict(sentences_score)
    accuracy = np.mean(predicted == y_score)
    print('Accuracy for score data: {:.4f}'.format(accuracy))

# print(metrics.classification_report(y_score, predicted, target_names=classes))
answer = classifier_fitted.predict(sentences_test)
df_answer = pd.DataFrame({
    'index': range(len(answer)),
    'type':  answer
})
df_answer.to_csv('answer.csv', index=False)
