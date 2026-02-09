import pandas as pd

# Usaremos un dataset de ejemplo con películas famosas
url = "https://raw.githubusercontent.com/kishan0725/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis/master/main_data.csv"

try:
    print("Descargando datos reales de películas...")
    df = pd.read_csv(url)
    df.to_csv('main_data.csv', index=False)
    print("¡Listo! main_data.csv ha sido creado con éxito.")
except Exception as e:
    print(f"Error al descargar: {e}")