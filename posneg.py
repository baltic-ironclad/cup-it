import os
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold

train_df = pd.read_csv(os.path.join('data', 'posneg', 'train_data.csv'))
test_df = pd.read_csv(os.path.join('data', 'posneg', 'test_data.csv'))

tf_idf = TfidfVectorizer(
    ngram_range=(1, 4), stop_words=stopwords.words('russian'),
    analyzer='word', max_df=0.8, min_df=10
)
tf_idf_model = tf_idf.fit(np.concatenate([train_df.text, test_df.text]))
train_tfidf_vec = tf_idf_model.transform(train_df.text)
test_tfidf_vec = tf_idf_model.transform(test_df.text)

# naive bayes
# classifier = MultinomialNB().fit(train_tfidf_vec, train_df.score)

# gradient descent
# classifier = SGDClassifier()
# classifier.fit(train_tfidf_vec, train_df.score)

# logistic regression
lm = LogisticRegression(penalty='l2', C=10)
lm_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100]
}
lm_search = GridSearchCV(
    estimator=lm, param_grid=lm_params,
    scoring='roc_auc', cv=StratifiedKFold(10),
    n_jobs=-1, verbose=1
)
lm_search_fitted = lm_search.fit(X=train_tfidf_vec, y=pd.factorize(train_df.score)[0])
prediction_scores = cross_val_score(estimator=lm_search_fitted.best_estimator_,
                                    X=train_tfidf_vec,
                                    y=pd.factorize(train_df.score)[0])
print(np.mean(prediction_scores))
prediction = lm_search_fitted.best_estimator_.predict_proba(test_tfidf_vec)[:, 0]

submission = pd.DataFrame({
    'index': range(0, len(prediction)),
    'score': prediction
})
submission.to_csv('posneg.csv', index=False)
