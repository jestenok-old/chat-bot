import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
import psycopg2
import pymorphy2
import config


conn = (psycopg2.connect(database=config.NAME,
                         user=config.USER,
                         password=config.PASSWORD,
                         host=config.HOST,
                         port=config.PORT))

sql_query = pd.read_sql_query('''select question, answer from questions limit 13000''', conn)

df = pd.DataFrame(sql_query, columns=['question', 'answer'])
conn.close()

stop = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
cv = CountVectorizer()
tfidf = TfidfVectorizer()


def step1(x):
    for i in x:
        a = str(i).lower()
        p = re.sub(r'[^а-я0-9]', ' ', a)
        print(p)


def text_normalization(text):
    text = str(text).lower()
    spl_char_text = re.sub(r'[^ а-я]', '', text)
    tokens = nltk.word_tokenize(spl_char_text)
    lema_words = []  # empty list
    for token in tokens:
        lema_token = morph.parse(token)[0].normal_form
        lema_words.append(lema_token)
    return " ".join(lema_words)


df['lemmatized_text'] = df['question'].apply(text_normalization)
X = cv.fit_transform(df['lemmatized_text']).toarray()
features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns=features)
df_simi = pd.DataFrame(df, columns=['answer', 'similarity_bow'])
df_simi_sort = df_simi.sort_values(by='similarity_bow', ascending=False)
threshold = 0.2
df_threshold = df_simi_sort[df_simi_sort['similarity_bow'] > threshold]
x_tfidf = tfidf.fit_transform(df['lemmatized_text']).toarray()
df_tfidf = pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names())


def stopword_(text):
    tag_list = nltk.word_tokenize(text)
    stop = stopwords.words('russian')
    lema_word = []
    for token in tag_list:
        if token in stop:
            continue
        lema_token = morph.parse(token)[0].normal_form
        lema_word.append(lema_token)
    return " ".join(lema_word)


def chat_bow(text):
    s = stopword_(text)
    lemma = text_normalization(s)
    bow = cv.transform([lemma]).toarray()
    cosine_value = 1 - pairwise_distances(df_bow, bow, metric='cosine')
    index_value = cosine_value.argmax()
    return df['answer'].loc[index_value]


def chat_tfidf(text):
    lemma = text_normalization(text)
    tf = tfidf.transform([lemma]).toarray()
    cos = 1 - pairwise_distances(df_tfidf, tf, metric='cosine')
    index_value = cos.argmax()
    return df['answer'].loc[index_value]
