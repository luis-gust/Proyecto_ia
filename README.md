# Recomendador de Películas con Análisis de Sentimiento

## Planteamiento del Proyecto

La idea principal de este proyecto es desarrollar una aplicación web que permita a los usuarios obtener recomendaciones de películas basadas en la similitud de contenido y, además, analizar el sentimiento de las reseñas de usuarios sobre dichas películas. El sistema combinará técnicas de procesamiento de lenguaje natural (NLP) y machine learning para ofrecer una experiencia interactiva y personalizada.

### Objetivo General

Crear una plataforma donde el usuario pueda:
- Buscar una película de su interés.
- Recibir recomendaciones de películas similares.
- Visualizar reseñas de usuarios y conocer el análisis automático de sentimiento (positivo o negativo) de cada reseña.

### Justificación

Actualmente, la cantidad de contenido audiovisual disponible es abrumadora. Un sistema que ayude a descubrir nuevas películas y que, además, analice automáticamente las opiniones de otros usuarios, puede mejorar significativamente la experiencia de búsqueda y selección de películas.

## ¿Cómo se ejecutará la idea?

1. **Interfaz Web**: Se desarrollará una aplicación web sencilla donde el usuario podrá ingresar el nombre de una película.
2. **Recomendación**: El sistema buscará películas similares utilizando técnicas de NLP para comparar descripciones, géneros y otros metadatos.
3. **Análisis de Sentimiento**: Se recopilarán reseñas de usuarios (por ejemplo, desde IMDB) y se aplicará un modelo de machine learning previamente entrenado para clasificar cada reseña como positiva o negativa.
4. **Presentación de Resultados**: El usuario verá una lista de películas recomendadas junto con las reseñas y su análisis de sentimiento.

## Ejecución esperada

- El usuario accede a la web y busca una película.
- El sistema muestra recomendaciones y reseñas analizadas.
- Todo el procesamiento de texto y predicción de sentimiento se realiza automáticamente en el backend.

## Tecnologías sugeridas

- Python (Flask para la web, scikit-learn para ML)
- HTML/CSS para la interfaz
- Pandas y Numpy para manejo de datos
- BeautifulSoup/Requests para scraping de reseñas

---

Este README describe la visión y el planteamiento inicial del proyecto. El desarrollo posterior incluirá la definición de la arquitectura, la recolección de datos, el entrenamiento de modelos y la implementación de la interfaz web.
