import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Crear main_data.csv (Mínimo requerido)
data = {
    'movie_title': ['avatar', 'titanic', 'star wars', 'the avengers'],
    'comb': ['action adventure fantasy james cameron', 'drama romance leonardo dicaprio', 'scifi space adventure george lucas', 'action hero marvel']
}
df = pd.DataFrame(data)
df.to_csv('main_data.csv', index=False)

# 2. Crear un modelo NLP y un Vectorizador de prueba
# Entrenamos con algo mínimo para que el archivo exista
textos = ["good movie", "bad movie", "excellent", "terrible"]
sentimientos = [1, 0, 1, 0] # 1=Good, 0=Bad

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)
clf = MultinomialNB()
clf.fit(X, sentimientos)

# Guardar los .pkl
pickle.dump(clf, open('nlp_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

print("Archivos base creados con éxito.")