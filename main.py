from utils import load_data
from content_based import build_content_model, recommend_content
from collaborative import build_collaborative_model
from evaluate import evaluate_rmse
from neural_net import NeuralCF, train_model, evaluate_model, build_mappings
from feedback import store_feedback, load_feedback

import pandas as pd

def main():
    # Load original data
    data, ratings, movies = load_data()

    # Load feedback data (may be empty initially)
    feedback = load_feedback()

    # Combine original ratings and feedback for training and evaluation
    all_ratings = pd.concat([ratings[['user_id', 'movie_id', 'rating']], feedback], ignore_index=True)

    print("Choose model to run:\n1. Content-Based\n2. Collaborative Filtering (SVD)\n3. Neural Collaborative Filtering (NCF)")
    choice = input("Enter 1, 2 or 3: ")

    if choice == '1':
        print("Building content-based model...")
        cosine_sim, indices = build_content_model(movies)

        title = input("Enter movie title (e.g., Star Wars (1977)): ")
        recs = recommend_content(title, cosine_sim, indices, movies)
        print(f"\nTop recommendations for '{title}':\n{recs}")

        # Ask for feedback on recommendations
        for rec_title in recs:
            give_feedback = input(f"Did you like '{rec_title}'? (yes/no): ").strip().lower()
            if give_feedback == 'yes':
                try:
                    user_id = int(input("Enter your User ID (1-943): "))
                    movie_id = movies[movies['title'] == rec_title]['movie_id'].values[0]
                    rating = float(input("Enter your rating (1-5): "))
                    store_feedback(user_id, movie_id, rating)
                    print("✅ Feedback saved successfully.")
                except Exception as e:
                    print(f"Error saving feedback: {e}")

    elif choice == '2':
        print("Building collaborative filtering model...")
        model, testset = build_collaborative_model(all_ratings)
        rmse = evaluate_rmse(model, testset)
        print(f"\nModel RMSE: {rmse:.4f}")

        try:
            user = int(input("Enter User ID (1–943): "))
            movie = int(input("Enter Movie ID (1–1682): "))
            prediction = model.predict(user, movie)
            print(f"Predicted rating for user {user} on movie {movie}: {prediction.est:.2f}")
        except Exception as e:
            print(f"Invalid input or prediction error: {e}")
            return

        feedback_input = input("Would you like to give feedback on this prediction? (yes/no): ").strip().lower()
        if feedback_input == 'yes':
            try:
                new_rating = float(input("Enter your rating for this movie (1-5): "))
                store_feedback(user, movie, new_rating)
                print("✅ Feedback saved successfully.")
            except Exception as e:
                print(f"Error saving feedback: {e}")

    elif choice == '3':
        print("Training Neural Collaborative Filtering model...")

        # Build user/movie index mappings from combined data
        user_ids = all_ratings['user_id'].unique()
        movie_ids = all_ratings['movie_id'].unique()
        user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

        num_users = len(user2idx)
        num_movies = len(movie2idx)

        # Filter ratings to only those users/movies in mapping
        train_data = pd.DataFrame({
            'user_idx': [user2idx[u] for u in all_ratings['user_id'] if u in user2idx],
            'movie_idx': [movie2idx[m] for m in all_ratings['movie_id'] if m in movie2idx],
            'rating': all_ratings['rating']
        })

        model = NeuralCF(num_users, num_movies, embedding_dim=50)
        train_model(model, train_data, epochs=5)
        rmse = evaluate_model(model, train_data)
        print(f"\nNeural Net RMSE: {rmse:.4f}")

        print("\nWould you like to provide feedback? (y/n)")
        if input().lower() == 'y':
            try:
                user_id = int(input("Enter your user ID (1–943): "))
                movie_id = int(input("Enter movie ID (1–1682): "))
                rating = float(input("Your rating (1–5): "))

                store_feedback(user_id, movie_id, rating)
                print("✅ Feedback saved successfully.")

                print("Would you like to retrain the neural model with feedback now? (y/n)")
                if input().lower() == 'y':
                    print("Retraining neural model with feedback...")

                    feedback = load_feedback()
                    all_ratings = pd.concat([ratings[['user_id', 'movie_id', 'rating']], feedback], ignore_index=True)

                    # Rebuild mappings (to capture any new users/movies)
                    user_ids = all_ratings['user_id'].unique()
                    movie_ids = all_ratings['movie_id'].unique()
                    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
                    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

                    num_users = len(user2idx)
                    num_movies = len(movie2idx)

                    train_data = pd.DataFrame({
                        'user_idx': [user2idx[u] for u in all_ratings['user_id'] if u in user2idx],
                        'movie_idx': [movie2idx[m] for m in all_ratings['movie_id'] if m in movie2idx],
                        'rating': all_ratings['rating']
                    })

                    model = NeuralCF(num_users, num_movies, embedding_dim=50)
                    train_model(model, train_data, epochs=5)
                    rmse = evaluate_model(model, train_data)
                    print(f"\n🔁 Retrained Neural Net RMSE: {rmse:.4f}")
            except Exception as e:
                print(f"Error during feedback/retraining: {e}")

    else:
        print("Invalid choice.")

if __name__ == '__main__':
    main()
