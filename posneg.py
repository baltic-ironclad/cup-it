import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

train_df = pd.read_csv(os.path.join('data', 'posneg', 'train_data.csv'))
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(train_df.text)
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

# naive bayes
# clf = MultinomialNB().fit(X_train_tf, train_df.text)
clf = SGDClassifier().fit(X_train_tf, train_df.text)

test_df = pd.read_csv(os.path.join('data', 'posneg', 'test_data.csv'))
X_test_counts = vectorizer.transform(test_df.text)
X_test_tf = tf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tf)
df_answer = pd.DataFrame({'type':  predicted})
df_answer.to_csv('posneg.csv', index=True)
