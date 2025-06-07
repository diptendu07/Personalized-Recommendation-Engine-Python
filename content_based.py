from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_content_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['title'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    return cosine_sim, indices

def recommend_content(title, cosine_sim, indices, data, top_n=5):
    title = title.lower().strip()
    
    # Try to match with existing titles (case-insensitive, partial match)
    matched_titles = [t for t in indices.index if title in t.lower()]
    
    if not matched_titles:
        print(f"No movie found matching: '{title}'")
        return []
    
    matched_title = matched_titles[0]  # Choose first match
    idx = indices[matched_title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices].tolist()
