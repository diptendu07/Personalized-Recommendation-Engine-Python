import pandas as pd
import os

FEEDBACK_FILE = 'data/feedback.csv'

def store_feedback(user_id, movie_id, rating):
    feedback = pd.DataFrame([[user_id, movie_id, rating]], columns=['user_id', 'movie_id', 'rating'])
    if not os.path.exists(FEEDBACK_FILE):
        feedback.to_csv(FEEDBACK_FILE, index=False)
    else:
        feedback.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
