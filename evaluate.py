from surprise import accuracy

def evaluate_rmse(model, testset):
    predictions = model.test(testset)
    return accuracy.rmse(predictions)
