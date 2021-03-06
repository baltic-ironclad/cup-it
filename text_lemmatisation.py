import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def sentences_lemmatisation(sentences, filename, types=None):
    for i in range(len(sentences)):
        sentences[i] = analyzer(sentences[i])

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            is_word = True
            for letter in sentences[i][j]:
                if ord(letter) < 1072 or ord(letter) > 1103 and ord(letter) != 1105:
                    sentences[i][j] = ''
                    is_word = False
                    break
            if is_word:
                sentences[i][j] = morph.parse(sentences[i][j])[0].normal_form
        sentences[i] = ' '.join(sentences[i])
        if i % 1000 == 0:
            print('Sentence', i, 'complete')
        
    if types is not None:
        df_answer = pd.DataFrame({
            'score': types,
            'text': sentences})
    else:
        df_answer = pd.DataFrame({'text': sentences})
    df_answer.to_csv(filename, index=True)


df = pd.read_csv(os.path.join('data', 'posneg', 'test_data.csv'), encoding='utf-8')

cv = CountVectorizer()
analyzer = cv.build_analyzer()
sentences_lemmatisation(df['text'].values + ' ' + df['title'].values,
                        os.path.join('data', 'posneg', 'test_norm.csv'))
                        # df['score'].values)
