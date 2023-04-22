import pickle
from sklearn.linear_model import LogisticRegression
import pandas
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
df = pandas.read_csv('spam.csv', encoding="ISO-8859-1")


en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
df['clean'] = df['v2'].apply(lambda x: ' '.join(
    [word for word in x.split() if word.lower() not in (sw_spacy)]))

count_vectorizer = CountVectorizer()
count = count_vectorizer.fit_transform(df['v2'])
Y = df['v2']

arr = df.values
label = np.delete(arr, [1, 2, 3, 4, 5], axis=1)
label = label.ravel()

x_train, x_test, y_train, y_test = train_test_split(
    count, label, test_size=0.2, random_state=42)
l_count = LogisticRegression()
l_count.fit(x_train, y_train)

# Saving model to disk
pickle.dump(l_count, open('mdl.pkl', 'wb'))