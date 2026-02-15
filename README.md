# 🎬 Sistema de Recomendación de Películas con Análisis de Sentimientos

## 📋 Introducción

**Autor:** Luis Gustavo Rodriguez

Este proyecto es una aplicación web inteligente que combina dos tecnologías avanzadas: un **motor de recomendación de películas basado en similitud de contenido** y un **clasificador de sentimientos mediante Machine Learning**. La aplicación utiliza el aprendizaje automático supervisado para analizar reseñas de películas en IMDB, clasificándolas como positivas o negativas, y proporciona recomendaciones de películas similares basadas en un análisis de contenido. El sistema está diseñado para ofrecerle al usuario una experiencia inteligente e interactiva, donde puede descubrir nuevas películas y comprender de manera cuantificada la opinión de la comunidad.

---

## 🏗️ Estructura y Funcionamiento Completo del Proyecto

### 1. **Arquitectura General del Sistema**

El proyecto se estructura en tres componentes principales:

```
┌─────────────────────────────────────────────────────────────┐
│                    APLICACIÓN WEB (Flask)                   │
│                      app.py (servidor)                      │
└─────────────────────────────────────────────────────────────┘
         ↑                           ↑                      ↑
         │                           │                      │
    ┌────────────────┐      ┌──────────────────┐   ┌──────────────────┐
    │  MODELOS IA    │      │  DATOS DE PELÍCULAS  │   │  SCRAPING WEB  │
    │ (ML Supervisado)      │  (CSV)           │   │ (IMDB Reviews)  │
    └────────────────┘      └──────────────────┘   └──────────────────┘
```

---

### 2. **Componentes Técnicos Detallados**

#### **A) Módulo de Entrenamiento: `train_sentiment.py`**

Este archivo implementa el corazón del sistema de **Clasificación de Sentimientos (Machine Learning Supervisado)**:

**Proceso de Entrenamiento:**

1. **Recopilación de Datos de Entrenamiento**
   - Se utiliza un dataset etiquetado con 20 reseñas de películas
   - Cada reseña tiene una etiqueta binaria: `1` (positiva/good) o `0` (negativa/bad)
   - Ejemplos: 
     - Positiva: *"I loved this movie, it was great"* → 1
     - Negativa: *"Terrible film, a waste of time"* → 0

2. **Vectorización con TF-IDF (Term Frequency-Inverse Document Frequency)**
   ```
   Texto: "I loved this movie, it was great"
   ↓
   [0.32, 0.51, 0.0, 0.45, ...] ← Vector numérico
   ```
   - **TF-IDF** transforma palabras en números que la IA entiende
   - Ignora palabras comunes como "the", "is", "a" (stop words)
   - Enfatiza palabras importantes como "loved", "great", "terrible", "waste"
   - **Fórmula:** `TF-IDF = (Frecuencia del término / Total de términos) × log(Total documentos / Documentos con término)`

3. **Entrenamiento con Naive Bayes Multinomial**
   - Algoritmo probabilístico ideal para clasificación de texto
   - Calcula la probabilidad de que un texto sea positivo o negativo
   - Funciona bajo la premisa de "independencia condicional" de características
   - **Fórmula de Bayes:** `P(sentimiento|texto) = P(texto|sentimiento) × P(sentimiento) / P(texto)`

4. **Exportación de Modelos (Serialización)**
   - `nlp_model.pkl`: Clasificador entrenado (Naive Bayes)
   - `tranform.pkl`: Vectorizador (TF-IDF) entrenado
   - Estos archivos permiten reutilizar el modelo sin reentrenamiento

**Código Clave:**
```python
# Vectorización
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])  # Matriz de características

# Entrenamiento
clf = MultinomialNB()
clf.fit(X, y)  # y = etiquetas (0 o 1)

# Guardado para reutilización
pickle.dump(clf, open('nlp_model.pkl', 'wb'))
```

---

#### **B) Obtención de Datos: `get_real_data.py`**

- Descarga un dataset CSV con información de películas desde GitHub
- Contiene campos: `movie_title`, `genre`, `overview`, `comb` (características combinadas)
- Usado por el motor de recomendación para calcular similitudes

---

#### **C) Motor Principal: `app.py`**

**Carga de Modelos Entrenados:**
```python
clf = pickle.load(open('nlp_model.pkl', 'rb'))        # Clasificador
vectorizer = pickle.load(open('tranform.pkl', 'rb'))  # Vectorizador
```

**Tres Funcionalidades Principales:**

**1. Motor de Recomendación - `rcmd(m)`**
- Calcula similitud de coseno entre películas
- Usa la matriz de características TF-IDF generada a partir del campo `comb`
- Retorna las 10 películas más similares a la ingresada

```python
# Similitud de Coseno
similarity = cosine_similarity(count_matrix)
# Rango: 0 a 1 (1 = idénticas, 0 = completamente diferentes)
```

**2. Ruta de Búsqueda - `/similarity` (POST)**
- Usuario ingresa nombre de película
- Sistema busca películas similares
- Retorna lista separada por `---`

**3. Ruta de Análisis Completo - `/recommend` (POST)**

**Pasos del proceso:**
```
Usuario selecciona película
         ↓
Sistema busca películas similares (recomendaciones)
         ↓
Web scraping: Descarga reseñas de IMDB
         ↓
Para cada reseña (máx 10):
   ├─ Vectorizar con TF-IDF (tranform.pkl)
   ├─ Clasificar con Naive Bayes (nlp_model.pkl)
   └─ Asignar sentimiento: 1→"Good" | 0→"Bad"
         ↓
Estadísticas:
   ├─ Total reseñas analizadas
   ├─ Reseñas positivas
   ├─ Reseñas negativas
   └─ Porcentaje positivo = (positivas/total) × 100
         ↓
Renderizar template con resultados
```

**Implementación del Análisis de Sentimientos en `app.py`:**
```python
# Dentro de la función recommend()
for reviews in soup_result[:10]:
    text = reviews.get_text()
    
    # 1. Transformar texto a vector numérico
    vector = vectorizer.transform([text])
    
    # 2. Predecir sentimiento (0 o 1)
    prediction = clf.predict(vector)
    
    # 3. Clasificar
    if prediction[0] == 1:
        sentiment = 'Good'
        reviews_stats["good"] += 1
    else:
        sentiment = 'Bad'
        reviews_stats["bad"] += 1
    
    reviews_stats["total"] += 1

# Cálculo de porcentaje
reviews_stats["percent"] = round((reviews_stats["good"] / reviews_stats["total"]) * 100)
```

---

#### **D) Interfaz Web: `templates/home.html`**

- Interfaz moderna con tema oscuro (inspirado en Netflix)
- Búsqueda de películas con autocomplete
- Visualización de:
  - Póster, título, fecha de lanzamiento, calificación, duración
  - Sinopsis de la película
  - **Veredicto de IA**: Porcentaje de reseñas positivas con barra de progreso
  - Reseñas individuales clasificadas (etiquetadas con colores)
  - Películas recomendadas en carrusel horizontal

---

### 3. **Flujo de Datos - Ejemplo Práctico**

**Usuario busca:** "The Matrix"

```
Entrada: "the matrix"
         ↓
Sistema normaliza a minúsculas: "the matrix"
         ↓
Busca en CSV: data['movie_title'] == "the matrix"
         ↓
Obtiene índice: i = 100
         ↓
Calcula similitud con todas las películas:
   similarity[100] = [0.85, 0.92, 0.45, 0.78, ...]
         ↓
Ordena de mayor a menor similitud
         ↓
Retorna top 10 (excluye "The Matrix" a sí misma)
         ↓
Resultado: ["Inception", "Dark City", "12 Monkeys", ...]
         ↓
Usuario selecciona "Inception"
         ↓
Web scraping: Obtiene 10 reseñas de IMDB
         ↓
ANÁLISIS DE SENTIMIENTOS:
   Reseña 1: "Brilliant and mind-bending masterpiece"
   └─ TF-IDF: [0.45, 0.32, ..., 0.78, ...]
   └─ Predicción: 1 (Good)
   
   Reseña 2: "Confusing and way too complicated"
   └─ TF-IDF: [0.52, ..., 0.31, ...]
   └─ Predicción: 0 (Bad)
         ↓
Estadísticas finales:
   - Total: 10 reseñas
   - Positivas: 8
   - Negativas: 2
   - Porcentaje: 80% positivo ✅
         ↓
Renderizar página con todos los datos
```

---

### 4. **Tecnologías y Librerías Utilizadas**

| Tecnología | Propósito |
|-----------|----------|
| **Flask** | Framework web para crear servidor local |
| **scikit-learn** | Algoritmos ML: TF-IDF, Naive Bayes, Cosine Similarity |
| **pandas** | Manipulación de datos CSV |
| **NumPy** | Cálculos numéricos y matrices |
| **BeautifulSoup4** | Web scraping de reseñas IMDB |
| **requests** | Descargar contenido web |
| **pickle** | Serializar/deserializar modelos |
| **LXML** | Parser HTML para BeautifulSoup |

---

### 5. **Características Clave del Machine Learning**

**A) Tipo de Aprendizaje: Supervisado**
- El modelo se entrena con datos etiquetados (reseñas + sentimiento)
- Aprende patrones de palabras positivas vs negativas
- Puede predecir el sentimiento de textos nuevos

**B) Algoritmo Naive Bayes Multinomial**
- Asume independencia entre características (palabras)
- Calcula probabilidades condicionales
- Ideal para textos cortos y clasificación binaria
- Tiempo de entrenamiento muy rápido

**C) Vectorización TF-IDF vs CountVectorizer**
- **TF-IDF** (en `train_sentiment.py`): Pondera importancia de palabras
  - Penaliza palabras muy comunes
  - Mejor para análisis de sentimientos
  
- **CountVectorizer** (en `app.py` para recomendaciones): Solo cuenta ocurrencias
  - Más simple y rápido
  - Suficiente para similitud de contenido

---

## 📦 Instalación y Configuración

### **Requisitos del Sistema**
- Python 3.8 o superior
- Conexión a Internet (para descargar datos y scraping)

### **Paso 1: Clonar o Descargar el Proyecto**
```bash
# Si tienes Git
git clone <url-del-repositorio>
cd Proyecto_ia

# O simplemente descarga los archivos manualmente
```

### **Paso 2: Instalar Dependencias**
```bash
# Navega a la carpeta del proyecto en PowerShell
cd D:\Doc\Universidad\Proyecto_ia

# Instala todas las librerías requeridas
pip install -r requirements.txt
```

### **Paso 3: Obtener Datos de Películas**
```bash
# Descarga el dataset CSV desde GitHub
python get_real_data.py
```
**Salida esperada:**
```
Descargando datos reales de películas...
¡Listo! main_data.csv ha sido creado con éxito.
```

### **Paso 4: Entrenar el Modelo de Sentimientos**
```bash
# Genera los modelos serializados (.pkl)
python train_sentiment.py
```
**Salida esperada:**
```
Cargando datos de entrenamiento...
Vectorizando textos con TF-IDF...
Entrenando el modelo Naive Bayes Multinomial...
Guardando archivos serializados (.pkl)...
------------------------------
¡ÉXITO! Se han generado 'nlp_model.pkl' y 'tranform.pkl'.
Ahora puedes ejecutar 'py app.py' para iniciar la web.
------------------------------
```

### **Paso 5: Ejecutar la Aplicación Web**
```bash
# Inicia el servidor Flask
python app.py
```
**Salida esperada:**
```
IA: Modelos de sentimiento cargados.
 * Running on http://127.0.0.1:5000
```

### **Paso 6: Acceder a la Aplicación**
- Abre tu navegador web
- Ve a: `http://localhost:5000/` o `http://127.0.0.1:5000/`
- Busca una película en inglés (ej: "The Matrix", "Inception", "Avatar")
- Haz clic en "Analizar" y espera los resultados

---

## 🎯 Cómo Usar la Aplicación

### **Interfaz Principal**
1. **Campo de Búsqueda:** Escribe el nombre de una película en inglés
2. **Botón Analizar:** Dispara la búsqueda de recomendaciones y análisis

### **Resultados Mostrados**
- **Póster de la película:** Imagen oficial
- **Información:** Título, fecha, calificación IMDB (0-10), duración
- **Sinopsis:** Descripción de la trama
- **Veredicto de IA:** Porcentaje de reseñas positivas (calculado por el modelo ML)
- **Reseñas Clasificadas:** Análisis individual de cada reseña con sentimiento
- **Películas Recomendadas:** Top 10 películas similares

---

## 🔬 Ejemplo de Análisis de Sentimientos

**Reseña Original:** *"This movie was absolutely brilliant! The cinematography was stunning and the actors delivered outstanding performances."*

**Proceso:**
```
1. Vectorización (TF-IDF):
   "absolutely" → 0.65 (palabra fuertemente positiva)
   "brilliant" → 0.72 (palabra clave positiva)
   "stunning" → 0.68 (adjetivo positivo)
   "outstanding" → 0.70 (superlativo positivo)
   
2. Clasificación Naive Bayes:
   P(Positivo|texto) = 0.94
   P(Negativo|texto) = 0.06
   
3. Predicción:
   0.94 > 0.06 → Sentimiento = POSITIVO (Good) ✅
```

---

## ✅ Conclusiones y Resumen

### **Logros del Proyecto**

Este sistema demuestra de manera práctica cómo el **Machine Learning Supervisado** puede ser aplicado a problemas reales:

1. **Inteligencia Artificial Funcional:** Modelo de clasificación de sentimientos entrenado que opera en tiempo real
2. **Análisis Predictivo:** Capaz de predecir sentimientos en textos nunca antes vistos (reseñas de IMDB en vivo)
3. **Experiencia de Usuario:** Interfaz web intuitiva que integra datos reales y análisis automático
4. **Integración de Tecnologías:** Combina web scraping, ML, bases de datos y frontend web

### **Conceptos Técnicos Reforzados**

- ✅ **Vectorización de Texto:** Transformación de palabras a números
- ✅ **Algoritmos de Clasificación:** Naive Bayes multinomial
- ✅ **Búsqueda de Similitud:** Cosine similarity para recomendaciones
- ✅ **Web Scraping:** Extracción automática de datos desde IMDB
- ✅ **Serialización de Modelos:** Persistencia de entrenamiento con pickle

### **Aplicaciones Futuras**

- Entrenar con dataset más grande (100,000+ reseñas) para mayor precisión
- Implementar redes neuronales (LSTM, BERT) para análisis más sofisticado
- Agregar análisis multilingüe (español, francés, etc.)
- Crear sistema de recomendación basado en perfiles de usuario
- Integrar base de datos SQL para almacenamiento persistente

---

## 📋 Estructura de Archivos Generados

Después de ejecutar los pasos de instalación, tu carpeta contendrá:

```
Proyecto_ia/
├── app.py                    # Servidor web principal
├── train_sentiment.py        # Entrena el modelo de sentimientos
├── get_real_data.py          # Descarga dataset de películas
├── requirements.txt          # Dependencias del proyecto
├── nlp_model.pkl            # ✅ Modelo Naive Bayes entrenado
├── tranform.pkl             # ✅ Vectorizador TF-IDF entrenado
├── main_data.csv            # ✅ Dataset de películas
├── README.md                # Este archivo
└── templates/
    └── home.html            # Interfaz web HTML/CSS
```

---

## 🛠️ Solución de Problemas

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Ejecuta `pip install -r requirements.txt` |
| `FileNotFoundError: nlp_model.pkl not found` | Ejecuta `python train_sentiment.py` primero |
| `FileNotFoundError: main_data.csv not found` | Ejecuta `python get_real_data.py` |
| Error de conexión a IMDB | Verifica tu conexión a Internet; IMDB puede bloquear requests frecuentes |
| Puerto 5000 en uso | Cambia el puerto en app.py: `app.run(debug=True, port=5001)` |

---

## 📞 Información del Autor

**Nombre:** Luis Gustavo Rodriguez

**Proyecto:** Sistema Inteligente de Recomendación de Películas con Análisis de Sentimientos

**Propósito:** Demostración práctica de Machine Learning Supervisado, NLP y desarrollo web

---

**Última actualización:** Febrero 2026

*Este proyecto es una herramienta educativa que demuestra la aplicación práctica de inteligencia artificial en casos de uso reales.*

