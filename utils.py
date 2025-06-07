import pandas as pd

def load_data():
    movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])
    ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    data = pd.merge(ratings, movies, on='movie_id')
    return data, ratings, movies

def get_id_mappings(ratings):
    user2idx = {uid: idx for idx, uid in enumerate(ratings['user_id'].unique())}
    movie2idx = {mid: idx for idx, mid in enumerate(ratings['movie_id'].unique())}
    return user2idx, movie2idx