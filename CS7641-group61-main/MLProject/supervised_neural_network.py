import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math

# take users with at least 10 positive interactions 
# leave one positive interaction out per user (for testing, not in training) 
# train one global supervised model (not per user) using user_id, spotify_id, and spotify features 
# include negatives via negative sampling since we don’t have explicit dislikes 
# for evaluation: for each test user, use the model to predict probability of like  
# grab 1000 unlabeled songs + the one left-out positive as candidates 
# rank them by predicted score and calculate hit rate@10 (if held-out song is in top 10) 
# unlabeled songs come from spotify features (songs the user hasn’t interacted with) 

# DATASET
class MusicDataset(Dataset):
    """Custom PyTorch dataset that encodes users, songs, and audio features."""
    def __init__(self, df, user2idx, song2idx, feature_cols):
        self.users = torch.tensor([user2idx[u] for u in df['user_id']], dtype=torch.long)
        self.songs = torch.tensor([song2idx[s] for s in df['spotify_id']], dtype=torch.long)
        self.labels = torch.tensor(df['liked'].values, dtype=torch.float32)
        self.audio = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.audio[idx], self.labels[idx]


# MODEL
class GlobalRecModel(nn.Module):
    """Hybrid recommendation model combining embeddings and audio features."""
    def __init__(self, n_users, n_items, user_dim=64, item_dim=64, audio_dim=4, hidden_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_dim)
        self.item_emb = nn.Embedding(n_items, item_dim)
        self.audio_proj = nn.Linear(audio_dim, 32)
        self.mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, users, items, audio):
        u = self.user_emb(users)
        i = self.item_emb(items)
        a = self.audio_proj(audio)
        x = torch.cat([u, i, a], dim=1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


# TRAINING
def train_global_model(df, feature_cols, epochs=3, batch_size=512, lr=1e-3,
                       device=None, save_path="./models/supervised_model.pt"):
    """
    Train the recommendation model and monitor loss across epochs.
    Early stopping is triggered when loss does not improve for several epochs.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Encode user and song IDs as integer indices
    user2idx = {u: i for i, u in enumerate(df['user_id'].unique())}
    song2idx = {s: i for i, s in enumerate(df['spotify_id'].unique())}

    model = GlobalRecModel(len(user2idx), len(song2idx), audio_dim=len(feature_cols))
    model.to(device)

    # Optionally resume training from a saved model
    # if os.path.exists(save_path):
    #     retrain = input(f"Found an existing model at {save_path}. Retrain? [y/n] ")
    #     if retrain.lower() == 'n':
    #         checkpoint = torch.load(save_path, map_location=device)
    #         model.load_state_dict(checkpoint)
    #         return model, user2idx, song2idx

    # Build dataset and dataloader
    dataset = MusicDataset(df, user2idx, song2idx, feature_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    patience, min_delta = 3, 1e-4
    best_loss, epochs_no_improve = float("inf"), 0

    print("Starting training...")

    for epoch in range(epochs):
        batch_losses, total_loss = [], 0.0
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for users, songs, audio, labels in progress_bar:
            users, songs, audio, labels = (
                users.to(device),
                songs.to(device),
                audio.to(device),
                labels.to(device),
            )
            preds = model(users, songs, audio)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(os.path.dirname(save_path), f"checkpoint_epoch_{epoch + 1}.pt"))

        # Early stopping check
        if avg_loss + min_delta < best_loss:
            best_loss, epochs_no_improve = avg_loss, 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch + 1} (no improvement in {patience} epochs).")
            break

    # Plot training loss
    epochs_range = range(1, len(epoch_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, epoch_losses, "o-", color="tab:blue", label="Average Loss")
    plt.title("Training Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs("./models", exist_ok=True)
    plt.savefig("./models/loss_curve.png")
    plt.close()

    print("Saved training loss plot to ./models/")
    return model, user2idx, song2idx


# EVALUATION
def evaluate(model, df, feature_cols, user2idx, song2idx,
             device=None, print_every=50, max_users=100, k=10):
    """
    Evaluate recommendation quality using:
    - HitRate@k
    - Precision@k
    - Recall@k
    - F1@k
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    all_users = list(df['user_id'].unique())
    users_to_eval = random.sample(all_users, min(len(all_users), max_users))
    all_songs = list(df['spotify_id'].unique())

    precision_list, recall_list, f1_list, hit_rates = [], [], [], []

    print(f"Evaluating {len(users_to_eval)} users...")

    for idx, user in enumerate(users_to_eval, 1):
        group = df[df['user_id'] == user]
        liked_songs = set(group[group['liked'] == 1]['spotify_id'].values)
        if len(liked_songs) < 2:
            continue

        # Select one positive song to hold out
        held_out = random.choice(list(liked_songs))
        negatives = [s for s in all_songs if s not in liked_songs]
        if len(negatives) < 999:
            continue
        neg_sample = random.sample(negatives, 999)
        candidates = [held_out] + neg_sample

        # Build candidate DataFrame with features
        cdf = pd.DataFrame({'user_id': user, 'spotify_id': candidates})
        cdf = cdf.merge(
            df.drop_duplicates('spotify_id')[['spotify_id'] + feature_cols],
            on='spotify_id', how='left'
        ).fillna(0)

        # Tensor conversion
        users = torch.tensor([user2idx[user]] * len(cdf), dtype=torch.long).to(device)
        songs = torch.tensor([song2idx[s] for s in cdf['spotify_id']], dtype=torch.long).to(device)
        audio = torch.tensor(cdf[feature_cols].values, dtype=torch.float32).to(device)

        with torch.no_grad():
            scores = model(users, songs, audio)

        # Get top-k items
        _, top_indices = torch.topk(scores, k=k)
        top_k_items = [candidates[i] for i in top_indices.tolist()]

        # Compute metrics for this user
        hits = len([item for item in top_k_items if item in liked_songs])
        precision = hits / k
        recall = hits / len(liked_songs) if len(liked_songs) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        hit = int(held_out in top_k_items)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        hit_rates.append(hit)

        if idx % print_every == 0:
            print(f"Processed {idx}/{len(users_to_eval)} users.")

    # Average metrics
    hit_rate_k = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    avg_prec = sum(precision_list) / len(precision_list) if precision_list else 0.0
    avg_rec = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    print("\n--- Recommendation Metrics ---")
    print(f"HitRate@{k}:   {hit_rate_k:.4f}")
    print(f"Precision@{k}: {avg_prec:.4f}")
    print(f"Recall@{k}:    {avg_rec:.4f}")
    print(f"F1@{k}:        {avg_f1:.4f}")

    return {
        "HitRate@k": hit_rate_k,
        "Precision@k": avg_prec,
        "Recall@k": avg_rec,
        "F1@k": avg_f1
    }