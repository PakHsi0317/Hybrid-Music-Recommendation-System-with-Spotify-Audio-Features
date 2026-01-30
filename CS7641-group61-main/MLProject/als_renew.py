import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
import argparse

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load positive interactions from CSV.
    Expects at least columns: user_id, spotify_id
    """
    df = pd.read_csv(csv_path)
    df = df[['user_id', 'spotify_id']].dropna().drop_duplicates()
    return df

def build_mappings(df: pd.DataFrame):
    """
    Map raw user/item IDs to contiguous indices.
    """
    users = df['user_id'].astype(str).unique()
    items = df['spotify_id'].astype(str).unique()

    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}

    return user2idx, item2idx, idx2user, idx2item

def df_to_sparse(df: pd.DataFrame, user2idx, item2idx) -> csr_matrix:
    """
    Build a sparse CSR matrix R (n_users x n_items)
    with 1 for each observed interaction.
    """
    rows = df['user_id'].astype(str).map(user2idx).values
    cols = df['spotify_id'].astype(str).map(item2idx).values
    data = np.ones(len(df), dtype=np.float32)

    R = coo_matrix(
        (data, (rows, cols)),
        shape=(len(user2idx), len(item2idx))
    ).tocsr()
    return R

def leave_k_out(R: csr_matrix, k: int = 1, seed: int = 42):
    """
    For each user, hold out k interacted items as test,
    remaining interactions go to train.
    Returns: train_matrix (CSR), test_items (dict u -> [items])
    """
    rng = np.random.default_rng(seed)
    n_users = R.shape[0]

    train_rows, train_cols = [], []
    test_items = {}

    for u in range(n_users):
        start, end = R.indptr[u], R.indptr[u + 1]
        items = R.indices[start:end]

        if len(items) == 0:
            continue

        k_eff = min(k, len(items))
        test = rng.choice(items, size=k_eff, replace=False)
        test_items[u] = list(test)

        for it in items:
            if it not in test:
                train_rows.append(u)
                train_cols.append(it)

    data = np.ones(len(train_rows), dtype=np.float32)
    train_mat = coo_matrix(
        (data, (train_rows, train_cols)),
        shape=R.shape
    ).tocsr()

    return train_mat, test_items

def als_train_with_metrics(R: csr_matrix,
                           test_items: dict,
                           factors: int = 128,
                           reg: float = 0.5,
                           alpha: float = 80.0,
                           iters: int = 20):
    """
    Optimized implicit ALS (Hu et al. 2008) with training loss logging
    AND HR/NDCG@5/10/20 tracked over iterations.
    R: CSR matrix of implicit feedback (1 for observed interactions).
    Returns:
        X, Y,
        loss_history (list[float]),
        hr_hist (dict[K] -> list over epochs),
        ndcg_hist (dict[K] -> list over epochs)
    """
    n_users, n_items = R.shape

    # preference p_ui = 1 for observed interactions
    P = R.copy()
    P.data = np.ones_like(P.data)

    # confidence c_ui = 1 + alpha * R_ui  (only on nonzeros)
    C = R.copy()
    C.data = 1.0 + alpha * C.data

    # latent factors initialization
    X = np.random.normal(0, 0.01, (n_users, factors))
    Y = np.random.normal(0, 0.01, (n_items, factors))

    loss_history = []

    # track HR/NDCG@K over epochs (K = 5,10,20)
    Ks = [5, 10, 20]
    hr_hist = {k: [] for k in Ks}
    ndcg_hist = {k: [] for k in Ks}

    def compute_train_loss():
        """
        Simple monitoring loss:
        on observed entries (u,i): (x_u^T y_i - 1)^2 + reg * (||X||^2 + ||Y||^2)
        """
        rows, cols = R.nonzero()
        preds = np.sum(X[rows] * Y[cols], axis=1)
        err = preds - 1.0
        mse = np.mean(err ** 2)
        reg_term = reg * (np.sum(X ** 2) + np.sum(Y ** 2)) / (X.size + Y.size)
        return mse + reg_term

    YtY = Y.T @ Y

    for epoch in range(iters):
        print(f"[ALS] Epoch {epoch + 1}/{iters}")

        for u in range(n_users):
            start, end = R.indptr[u], R.indptr[u + 1]
            items = R.indices[start:end]
            if len(items) == 0:
                continue

            Cu = C.data[start:end]     
            Pu = P.data[start:end]        
            Y_u = Y[items]            

            A = YtY + (Y_u.T * (Cu - 1.0)) @ Y_u + reg * np.eye(factors)
            b = (Y_u.T * Cu) @ Pu

            X[u] = np.linalg.solve(A, b)

        XtX = X.T @ X

        for i in range(n_items):

            col = R[:, i]
            users = col.indices
            if len(users) == 0:
                continue

            Ci = C[:, i].data   
            Pi = P[:, i].data  
            X_i = X[users]

            A = XtX + (X_i.T * (Ci - 1.0)) @ X_i + reg * np.eye(factors)
            b = (X_i.T * Ci) @ Pi

            Y[i] = np.linalg.solve(A, b)


        YtY = Y.T @ Y


        loss = compute_train_loss()
        loss_history.append(loss)
        print(f"    train loss: {loss:.6f}")

        # --------- compute HR/NDCG@K for this epoch ----------

        from math import inf
        scores = X @ Y.T
        scores[R.nonzero()] = -inf
        recs_epoch = np.argsort(-scores, axis=1)[:, :20]

        for k in Ks:
            hr_val = hit_rate_at_k(recs_epoch, test_items, k)
            ndcg_val = ndcg_at_k(recs_epoch, test_items, k)
            hr_hist[k].append(hr_val)
            ndcg_hist[k].append(ndcg_val)

    return X, Y, loss_history, hr_hist, ndcg_hist

def recommend_all(X: np.ndarray,
                  Y: np.ndarray,
                  train_mat: csr_matrix,
                  topk: int = 50):
    """
    Compute top-k recommendation list for each user.
    Mask out training interactions.
    Returns: recs (n_users x topk) numpy array of item indices.
    """
    scores = X @ Y.T 
    scores[train_mat.nonzero()] = -np.inf 
    recs = np.argsort(-scores, axis=1)[:, :topk]
    return recs

def hit_rate_at_k(recs, test_items, k: int) -> float:
    hits, users = 0, 0
    for u, items in test_items.items():
        users += 1
        topk = set(recs[u][:k])
        if any(t in topk for t in items):
            hits += 1
    return hits / max(users, 1)

def ndcg_at_k(recs, test_items, k: int) -> float:
    def dcg(rel):
        return np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))

    total, users = 0.0, 0
    for u, items in test_items.items():
        users += 1
        top = recs[u][:k]
        rel = np.array([1.0 if it in items else 0.0 for it in top])
        ideal = sorted(rel, reverse=True)
        total += dcg(rel) / (dcg(ideal) if dcg(ideal) != 0 else 1.0)
    return total / max(users, 1)

def mrr_at_k(recs, test_items, k: int) -> float:
    total_rr, users = 0.0, 0
    for u, items in test_items.items():
        users += 1
        top = recs[u][:k]
        rr = 0.0
        for t in items:
            if t in top:
                rr = 1.0 / (np.where(top == t)[0][0] + 1)
                break
        total_rr += rr
    return total_rr / max(users, 1)

def plot_als_training_curves(loss_history, hr_hist, ndcg_hist,
                             out_path: str = "als_training_curves.png"):

    epochs = np.arange(1, len(loss_history) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.3)

    ax = axes[0]
    ax.plot(epochs, loss_history)
    ax.set_title("Training Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax = axes[1]
    for k, vals in hr_hist.items():
        ax.plot(epochs, vals, label=f"HR@{k}")
    ax.set_title("HR Metrics Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("HR Score")
    ax.legend()

    ax = axes[2]
    for k, vals in ndcg_hist.items():
        ax.plot(epochs, vals, label=f"NDCG@{k}")
    ax.set_title("NDCG Metrics Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NDCG Score")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Plot saved] ALS training curves → {out_path}")

def plot_metrics(ks, hr, ndcg, mrr, out_path: str = "als_metrics_optimized.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, hr, label="HitRate@K")
    plt.plot(ks, ndcg, label="NDCG@K")
    plt.plot(ks, mrr, label="MRR@K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title("Optimized ALS Recommendation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Plot saved] metrics → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimized Implicit ALS Recommender")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="processed_music_data_no_negatives.csv",
        help="Path to processed_music_data_no_negatives.csv"
    )
    parser.add_argument("--leave_k", type=int, default=1, help="leave-k-out per user")
    parser.add_argument("--factors", type=int, default=128, help="latent dimension")
    parser.add_argument("--reg", type=float, default=0.5, help="regularization strength")
    parser.add_argument("--alpha", type=float, default=80.0, help="implicit feedback alpha")
    parser.add_argument("--iters", type=int, default=20, help="ALS iterations")
    args = parser.parse_args()

    print(f"[Load] reading data from: {args.csv_path}")
    df = load_data(args.csv_path)

    user2idx, item2idx, idx2user, idx2item = build_mappings(df)
    R = df_to_sparse(df, user2idx, item2idx)
    print(f"[Info] users: {len(user2idx)}, items: {len(item2idx)}, interactions: {R.nnz}")

    train_mat, test_items = leave_k_out(R, k=args.leave_k)
    print("[Split] train interactions:", train_mat.nnz, "| test users:", len(test_items))

    print("[Train] Optimized ALS with training metrics...")
    X, Y, loss_history, hr_hist, ndcg_hist = als_train_with_metrics(
        train_mat,
        test_items,
        factors=args.factors,
        reg=args.reg,
        alpha=args.alpha,
        iters=args.iters
    )

    plot_als_training_curves(loss_history, hr_hist, ndcg_hist)

    recs = recommend_all(X, Y, train_mat, topk=50)
    ks = list(range(1, 21))
    hr = [hit_rate_at_k(recs, test_items, k) for k in ks]
    ndcg = [ndcg_at_k(recs, test_items, k) for k in ks]
    mrr = [mrr_at_k(recs, test_items, k) for k in ks]

    plot_metrics(ks, hr, ndcg, mrr)

    # Print final metrics only at K = 5, 10, 20
    for k in [5, 10, 20]:
        print(f"K={k:2d} | HR={hit_rate_at_k(recs, test_items, k):.6f} "
              f"NDCG={ndcg_at_k(recs, test_items, k):.6f} "
              f"MRR={mrr_at_k(recs, test_items, k):.6f}")


if __name__ == "__main__":
    main()
