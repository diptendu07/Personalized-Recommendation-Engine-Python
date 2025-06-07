from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

def build_collaborative_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    model = SVD()
    model.fit(trainset)
    return model, testset
