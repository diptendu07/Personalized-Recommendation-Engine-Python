# 🎬 Personalized Recommendation Engine

A modular movie recommendation system built using the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/). Supports three major recommendation strategies:

- **Content-Based Filtering**
- **Collaborative Filtering (SVD)**
- **Neural Collaborative Filtering (NCF)**

Includes dynamic user feedback integration and real-time retraining.

---

## 📁 Project Structure

```
recommendation_engine/
├── data/ # Raw and feedback data
│ ├── u.data # Ratings
│ ├── u.item # Movie metadata
│ └── feedback.csv # Dynamic feedback storage
├── content_based.py # Content-based recommendation engine
├── collaborative.py # Collaborative filtering using SVD
├── neural_net.py # Neural Collaborative Filtering model
├── evaluate.py # Evaluation utilities (e.g., RMSE)
├── feedback.py # User feedback capture & integration
├── utils.py # Data loading and preprocessing
├── main.py # CLI interface for model selection
├── requirements.txt
└── README.md
```

---

## 🧠 Core Features

### 1️⃣ Content-Based Filtering (`content_based.py`)
- Recommends movies similar to a given movie based on **TF-IDF vectorization** of titles.
- Uses **cosine similarity** for scoring.
- Supports partial and case-insensitive search.

### 2️⃣ Collaborative Filtering (SVD) (`collaborative.py`)
- Implements **matrix factorization** using the `Surprise` library.
- Learns latent user-item features and predicts unseen ratings.
- Evaluated using **RMSE**.

### 3️⃣ Neural Collaborative Filtering (NCF) (`neural_net.py`)
- Deep learning model using **PyTorch** with user/item embeddings.
- Learns non-linear user-item interaction patterns via dense layers.
- Supports **dynamic retraining** with feedback data.

### 🧪 Evaluation (`evaluate.py`)
- Computes **Root Mean Squared Error (RMSE)** on test data.
- Used to benchmark SVD and Neural models.

### 💬 Feedback System (`feedback.py`)
- Collects real-time feedback from users.
- Stores ratings in `feedback.csv`.
- Enables **adaptive retraining** of NCF model.

---

## 📂 File Descriptions
```
| Filename            | Purpose                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| `content_based.py`  | Implements Content-Based Filtering using TF-IDF vectorization and cosine similarity.   |
| `collaborative.py`  | Implements Collaborative Filtering with SVD matrix factorization via Surprise library. |
| `neural_net.py`     | Implements Neural Collaborative Filtering (NCF) with embeddings and MLP in PyTorch.    |
| `evaluate.py`       | Evaluation utilities, primarily RMSE calculation to assess prediction accuracy.        |
| `feedback.py`       | Collects and stores user feedback; integrates feedback for dynamic retraining.         |
| `utils.py`          | Data loading, preprocessing, and utility functions used across models.                 |
| `main.py`           | Command-line interface for model selection, running, and retraining workflow.          |
| `requirements.txt`  | Lists all Python package dependencies for the project.                                 |
| `data/u.data`       | Original user-movie ratings from the MovieLens 100k dataset.                           |
| `data/u.item`       | Movie metadata file from MovieLens dataset.                                            |
| `data/feedback.csv` | Stores dynamic user feedback ratings for retraining models.                            |
```

## 🚀 How to Run


### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Activate Python Virtual Environment
(Windows example)
```
.\venv\Scripts\activate
```

### 3. Run the Application
```
python main.py
```

### 4. Choose a Recommendation Method
```
1 → Content-Based (Enter movie title)

2 → SVD (Enter user ID and movie ID)

3 → NCF (Trains model, then prompts for user ID and movie ID)
```
---

## ✅ Example Workflow
### Case 1: Content-Based Filtering
Input: "Star Wars (1977)"

Output: Similar movies like The Empire Strikes Back, Return of the Jedi, etc.

### Case 2: Collaborative Filtering (SVD)
Input: User ID: 100, Movie ID: 50

Output: Predicted rating → e.g., 4.2/5

### Case 3: Neural Collaborative Filtering (NCF)
Input: Run training (5 epochs), then:

User ID: 100, Movie ID: 50

Output: Predicted rating → e.g., 4.3/5

User prompted for feedback → Rating saved and model can be retrained.

---

## 📦 Dependencies
### All required libraries are listed in requirements.txt:
```
pandas
numpy
scikit-learn
scipy
surprise
torch
```
---

## 📊 Evaluation Metrics
```
| Model                      | Evaluation Metric    |
| -------------------------- | -------------------- |
| Content-Based Filtering    | Similarity Score     |
| Collaborative Filtering    | RMSE                 |
| Neural Collaborative (NCF) | RMSE + Feedback Loop |
```
---

## 🧠 Future Enhancements

Include genres and full movie metadata
Sentiment analysis from descriptions or reviews
Web or REST API interface
Precision/Recall evaluation

---

## 📌 Credits
### Dataset: MovieLens 100k

### Libraries: Surprise, PyTorch, scikit-learn

---

## Project Requirements

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