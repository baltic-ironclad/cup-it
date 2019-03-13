import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

train_df = pd.read_csv(os.path.join('data', 'posneg', 'train_data.csv'))
test_df = pd.read_csv(os.path.join('data', 'posneg', 'test_data.csv'))

vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train_counts = vectorizer.fit_transform(train_df.text.values)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# naive bayes
# clf = MultinomialNB().fit(X_train_tf, train_df.text)
classifier = SGDClassifier()
classifier.fit(X_train_tfidf, train_df.text.values)

X_test_counts = vectorizer.transform(test_df.text.values)
X_test_tf = tfidf_transformer.transform(X_test_counts)

predicted = classifier.predict(X_test_tf)
df_answer = pd.DataFrame({'type':  predicted})
df_answer.to_csv('posneg.csv', index=True)
