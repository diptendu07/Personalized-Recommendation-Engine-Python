# 🎬 Personalized Recommendation Engine

This project implements a movie recommendation system using two core approaches:
- **Content-Based Filtering**
- **Collaborative Filtering (SVD)**

The engine is built on the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/) and provides personalized recommendations based on user preferences, movie content, and collaborative patterns among users.

---

## 📁 Project Structure

```
recommendation_engine/
├── data/
│   ├── u.data
│   ├── u.item
│   └── feedback.csv          
├── content_based.py
├── collaborative.py
├── evaluate.py
├── neural_net.py             
├── feedback.py               
├── utils.py
├── main.py
├── README.md
└── requirements.txt

```

---

## 💡 Approach & Models

### 1. 📚 Content-Based Filtering (`content_based.py`)

This method recommends movies **similar to a movie the user likes**, based on movie metadata such as titles.

#### Technique:
- **TF-IDF Vectorization**: Converts movie titles to vectors using `TfidfVectorizer`, capturing keyword importance while removing stopwords.
- **Cosine Similarity**: Computes similarity scores between movie vectors to find the most similar titles.
- **Title Matching**: Supports case-insensitive, partial matching to handle user input flexibly.

#### Sentiment-style Filtering:
While not using textual sentiment explicitly, TF-IDF acts as a **proxy to emotional tone** via keywords (e.g., *Love*, *Horror*, *Action*) — enabling a **semantic similarity** based filtering process.

#### Output:
Recommends `Top-N` similar movies to the one specified by the user.

---

### 2. 👥 Collaborative Filtering - SVD (`collaborative.py`)

This model recommends movies by learning **latent factors** that influence user preferences, using the **Singular Value Decomposition (SVD)** technique.

#### Technique:
- Uses the `Surprise` library to apply SVD on the user-movie ratings matrix.
- Learns patterns between users and items from historical data.
- Can predict how much a user would rate a specific unseen movie.

#### Evaluation:
- RMSE is used to evaluate prediction accuracy on a test set.
- Users can input a user ID and movie ID to receive a predicted rating.

#### Strength:
Captures **collaborative sentiment** — if users with similar tastes liked certain movies, the model will suggest those to the current user.

---

## 🧪 Evaluation (`evaluate.py`)

The system uses **Root Mean Squared Error (RMSE)** to evaluate the collaborative model's accuracy:

- Lower RMSE → Better predictions
- Implemented using the `accuracy` module from `surprise`

---

## ⚙️ How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt

2. Activate Python Environment (optional)
# For example, on Windows
.\venv\Scripts\activate

3. Run the Engine
python main.py

Choose:

1 for Content-Based Filtering (enter full or partial movie name)

2 for Collaborative Filtering (SVD)

Enter a user ID between 1 and 943

Enter a movie ID between 1 and 1680

---

📦 Dependencies
Listed in requirements.txt:

pandas

scikit-learn

scipy

numpy

surprise

---

✅ Example Workflow
Case 1: Content-Based Filtering
User selects 1 and enters "Star Wars (1977)".
The engine returns similar movies like The Empire Strikes Back, Return of the Jedi, etc.

Case 2: Collaborative Filtering
User selects 2, inputs user ID 100 and movie ID 50.
The engine predicts a rating like 4.2/5, helping the user decide if they might enjoy it.

---

🤖 3. Neural Collaborative Filtering (NCF) (neural_net.py)
This advanced recommendation method uses a Neural Network architecture to model complex user-item interactions beyond traditional matrix factorization.

Key Features:
Embedding layers for users and movies capture latent features.

Multi-layer perceptron learns nonlinear relationships between user preferences and item attributes.

Supports dynamic training with both original ratings and user feedback data.

Benefits:
More flexible than classical SVD-based collaborative filtering.

Capable of capturing subtle patterns in user behavior for better personalization.
---

💬 User Feedback & Dynamic Model Retraining (feedback.py & updates in main.py)
This project supports collecting explicit user feedback on recommendations and integrates it into the training process to improve recommendation quality over time.

Functionality:
Users can provide ratings on recommended movies directly after receiving recommendations.

Feedback is stored persistently in data/feedback.csv.

Feedback data is combined with original ratings to retrain models dynamically, especially the Neural Collaborative Filtering model.

Retraining can be triggered interactively within the command-line interface, allowing the system to adapt to evolving user preferences in near real-time.

Advantages:
Enables continuous learning and model improvement.

Personalizes recommendations more closely to users' current tastes.

Provides a practical feedback loop for real-world recommendation systems.
---

```
| Requirement                                                                           | Status in Your Project | Comments                                                                                                                             |
| ------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Use publicly available dataset (MovieLens)**                                        | ✅ Completed            | Using MovieLens 100k dataset loaded and cleaned in `utils.py` and elsewhere.                                                         |
| **Data cleansing**                                                                    | ✅ Completed            | Data is loaded and processed for modeling; duplicates and missing values handled.                                                    |
| **Build Content-Based Filtering**                                                     | ✅ Completed            | Implemented using TF-IDF vectorization of movie titles and cosine similarity (`content_based.py`).                                   |
| **Build Collaborative Filtering (matrix factorization/SVD)**                          | ✅ Completed            | Implemented using Surprise library SVD (`collaborative.py`).                                                                         |
| **Evaluate and compare models using metrics like Precision, Recall, or RMSE**         | ✅ Partially Completed  | RMSE evaluation implemented (`evaluate.py`); Precision/Recall not currently included but RMSE suffices for rating prediction models. |
| **Experiment with Neural Network-based recommendations (Embeddings or Autoencoders)** | ✅ Completed            | Neural Collaborative Filtering implemented with embeddings and multi-layer perceptron (`neural_net.py`).                             |
| **Allow users to give feedback on recommendations**                                   | ✅ Completed            | Feedback functionality implemented (`feedback.py`), feedback stored and used.                                                        |
| **Retrain models dynamically with feedback**                                          | ✅ Completed            | Users can retrain the Neural CF model interactively with feedback (`main.py`).                                                       |
```

🧠 Conclusion
This system combines semantic similarity (TF-IDF) and behavioral patterns (SVD) to deliver personalized movie recommendations. It is modular, extensible, and can be enhanced with:

Genre and keyword metadata

Full-text description analysis

Sentiment analysis from reviews

Web or API interface

---

