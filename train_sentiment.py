import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    # 1. Datos de entrenamiento (Dataset de aprendizaje)
    # Estas frases enseñan a la IA qué palabras tienen carga positiva o negativa
    print("Cargando datos de entrenamiento...")
    data = {
        'review': [
            'I loved this movie, it was great', 'Terrible film, a waste of time',
            'Excellent acting and plot', 'I hated every minute of it',
            'A masterpiece of modern cinema', 'The worst movie I have ever seen',
            'Highly recommended for everyone', 'Boring and predictable',
            'Amazing visuals and sound', 'I did not like the ending at all',
            'One of my favorite movies now', 'Total garbage, dont watch it',
            'The story was beautiful', 'Waste of money and breath',
            'Brilliant performance by the lead', 'Simply awful and annoying',
            'Best movie ever', 'So bad, I fell asleep',
            'I really enjoyed it', 'Do not waste your time'
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)

    # 2. PROCESAMIENTO (NLP): TF-IDF
    # TF-IDF ayuda a la IA a ignorar palabras comunes (the, is) y enfocarse en las importantes (amazing, waste).
    print("Vectorizando textos con TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']

    # 3. ENTRENAMIENTO (Machine Learning): Naive Bayes
    # Es un modelo probabilístico ideal para clasificación de texto.
    print("Entrenando el modelo Naive Bayes Multinomial...")
    clf = MultinomialNB()
    clf.fit(X, y)

    # 4. EXPORTACIÓN DE MODELOS (.pkl)
    # Guardamos el 'cerebro' (clf) y el 'traductor' (vectorizer) para usarlos en app.py
    print("Guardando archivos serializados (.pkl)...")
    pickle.dump(clf, open('nlp_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

    print("-" * 30)
    print("¡ÉXITO! Se han generado 'nlp_model.pkl' y 'tranform.pkl'.")
    print("Ahora puedes ejecutar 'py app.py' para iniciar la web.")
    print("-" * 30)

if __name__ == "__main__":
    train_model()