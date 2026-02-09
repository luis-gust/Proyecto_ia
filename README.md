# Movie AI: Sistema de Recomendación y Análisis de Sentimiento con IA

## 1. Introducción
**Movie AI** es una solución tecnológica diseñada para transformar la interacción del usuario con el cine. No es solo un motor de búsqueda; es una plataforma que integra **Procesamiento de Lenguaje Natural (NLP)** y **Aprendizaje Automático (Machine Learning)** para ofrecer una experiencia doble:
1.  **Recomendación Inteligente:** Encuentra películas similares basándose en el "ADN" del contenido (actores, directores, géneros).
2.  **Veredicto de la Crítica (IA):** Analiza en tiempo real las reseñas de IMDB para determinar, mediante un modelo predictivo, el porcentaje de aprobación real de la audiencia.

---

## 2. Desarrollo Técnico: El Corazón de la IA

El proyecto implementa dos tipos de inteligencia artificial para resolver problemas distintos.

### A. Clasificación de Sentimientos (Machine Learning Supervisado)
Para entender si una reseña es positiva o negativa, la IA pasa por un proceso de aprendizaje en el archivo `train_sentiment.py`.

#### Paso a paso del aprendizaje:
1.  **Vectorización (TF-IDF):** Las palabras se convierten en números. Utilizamos `TfidfVectorizer` para que la IA entienda qué palabras son clave (ej. "Masterpiece", "Waste") y cuáles son irrelevantes (ej. "the", "a").
    ```python
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    ```


2.  **Entrenamiento (Multinomial Naive Bayes):** Es un modelo probabilístico que calcula la probabilidad de que una frase sea "Buena" basándose en la frecuencia de palabras positivas aprendidas.
    ```python
    clf = MultinomialNB()
    clf.fit(X, y) # La IA asocia patrones de texto con etiquetas (1=Good, 0=Bad)
    ```


3.  **Inferencia en Tiempo Real:** En `app.py`, cuando el sistema hace el *scraping* de IMDB, cada reseña nueva pasa por el modelo entrenado:
    ```python
    vector = vectorizer.transform([text_imdb]) # Traduce reseña a números
    prediction = clf.predict(vector)           # La IA da el veredicto (0 o 1)
    ```

### B. Motor de Recomendación (Similitud del Coseno)
Para recomendar, la IA trata a cada película como un punto en un mapa de miles de dimensiones. 
* Se crea una "combinación" de metadatos (Género + Actores + Director).
* Se utiliza la **Similitud del Coseno** para medir la distancia entre estos puntos. Cuanto más pequeño es el ángulo entre dos vectores, más parecidas son las películas.

```python
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])
similarity = cosine_similarity(count_matrix)