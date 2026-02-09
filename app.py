import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bs4 as bs
import pickle
import requests

# --- 1. CARGA DE MODELOS DE IA ---
try:
    clf = pickle.load(open('nlp_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tranform.pkl', 'rb'))
    print("IA: Modelos de sentimiento cargados.")
except Exception as e:
    print(f"IA Error: No se pudieron cargar los archivos .pkl. {e}")


# --- 2. MOTOR DE RECOMENDACIÓN ---
def init_recommendation_engine():
    try:
        data = pd.read_csv('main_data.csv')
        data['movie_title'] = data['movie_title'].str.lower()
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'])
        similarity = cosine_similarity(count_matrix)
        return data, similarity
    except Exception as e:
        print(f"Recomendador Error: {e}")
        return None, None


data, similarity = init_recommendation_engine()


def rcmd(m):
    m = m.lower()
    try:
        if data is None or m not in data['movie_title'].unique():
            return None
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        return [data['movie_title'][a[0]] for a in lst]
    except:
        return None


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form['name']
    rc = rcmd(movie)
    if rc is None: return "error"
    return "---".join(rc)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        title = request.form['title']
        imdb_id = request.form['imdb_id']
        poster = request.form['poster']
        overview = request.form['overview']
        vote_average = request.form['rating']
        release_date = request.form['release_date']
        runtime = request.form['runtime']
        rec_movies = request.form.get('rec_movies', "").split('---')
        rec_posters = request.form.get('rec_posters', "").split('---')

        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_movies)) if i < len(rec_posters)}

        # --- 3. SCRAPING + ESTADÍSTICAS DE IA ---
        movie_reviews = {}
        reviews_stats = {"good": 0, "bad": 0, "total": 0, "percent": 0}

        try:
            url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            soup = bs.BeautifulSoup(response.text, 'lxml')
            soup_result = soup.find_all("div", {"class": "text show-more__control"})

            if not soup_result:
                soup_result = soup.select(".ipc-html-content-inner-div")

            for reviews in soup_result[:10]:  # Analizamos hasta 10 reseñas para el promedio
                text = reviews.get_text()
                if text:
                    vector = vectorizer.transform([text])
                    prediction = clf.predict(vector)

                    if prediction[0] == 1:
                        sentiment = 'Good'
                        reviews_stats["good"] += 1
                    else:
                        sentiment = 'Bad'
                        reviews_stats["bad"] += 1

                    reviews_stats["total"] += 1
                    movie_reviews[text] = sentiment

            if reviews_stats["total"] > 0:
                reviews_stats["percent"] = round((reviews_stats["good"] / reviews_stats["total"]) * 100)

        except Exception as e:
            print(f"Scraping Error: {e}")

        return render_template('home.html', title=title, poster=poster, overview=overview,
                               vote_average=vote_average, release_date=release_date,
                               runtime=runtime, movie_cards=movie_cards,
                               movie_reviews=movie_reviews, reviews_stats=reviews_stats)

    except Exception as e:
        print(f"General Error: {e}")
        return render_template('home.html', error_msg="Error al procesar la recomendación.")


if __name__ == '__main__':
    app.run(debug=True)