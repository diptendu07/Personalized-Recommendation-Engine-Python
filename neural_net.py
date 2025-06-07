import torch
import torch.nn as nn
import torch.optim as optim

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        x = torch.cat([u, i], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_mappings(ratings):
    user2idx = {user_id: idx for idx, user_id in enumerate(ratings['user_id'].unique())}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(ratings['movie_id'].unique())}
    return user2idx, movie2idx

def train_model(model, train_data, epochs=5, lr=0.005):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for _, row in train_data.iterrows():
            user = torch.tensor([row['user_idx']], dtype=torch.long)    # <-- updated dtype here
            item = torch.tensor([row['movie_idx']], dtype=torch.long)   # <-- updated dtype here
            rating = torch.tensor([[row['rating']]], dtype=torch.float32)  # shape (1,1)

            pred = model(user, item)  # output shape (1,1)
            loss = criterion(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")

def evaluate_model(model, test_data):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user = torch.tensor([row['user_idx']], dtype=torch.long)   # <-- updated dtype here
            item = torch.tensor([row['movie_idx']], dtype=torch.long)  # <-- updated dtype here
            rating = torch.tensor([[row['rating']]], dtype=torch.float32)  # shape (1,1)
            pred = model(user, item)
            loss = criterion(pred, rating)
            total_loss += loss.item()

    rmse = (total_loss / len(test_data)) ** 0.5
    return rmse
