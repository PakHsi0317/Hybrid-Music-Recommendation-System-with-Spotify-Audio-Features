#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict
from pathlib import Path

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Read positive feedback (liked=1) interactions between users and items (user_id, spotify_id) from a CSV file, 
#and return a DataFrame containing only three columns of positive feedback.
def load_interactions(csv_path: str, assume_liked_if_missing: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    base_cols = {'user_id', 'spotify_id'}
    if not base_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {base_cols}, got {set(df.columns)}")

    if 'liked' in df.columns:
        df_pos = df[df['liked'] == 1].copy()
    else:
        if assume_liked_if_missing:
            df_pos = df.copy(); df_pos['liked'] = 1
        else:
            raise ValueError("CSV has no 'liked' column; pass --assume_liked_if_missing to treat all rows as positives.")

    df_pos = df_pos[['user_id', 'spotify_id', 'liked']].dropna().drop_duplicates()
    if df_pos.empty:
        raise ValueError("No positive interactions found.")
    return df_pos

#Map the string user/item ID to a consecutive integer index (bidirectional).
def build_mappings(df_pos: pd.DataFrame):
    users = df_pos['user_id'].astype(str).unique()
    items = df_pos['spotify_id'].astype(str).unique()
    user_to_idx = {u:i for i,u in enumerate(users)}
    item_to_idx = {it:i for i,it in enumerate(items)}
    idx_to_user = {i:u for u,i in user_to_idx.items()}
    idx_to_item = {i:it for it,i in item_to_idx.items()}
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item

#Convert the positive feedback DataFrame into a sparse interaction matrix (CSR) with 
# shape [n_users, n_items], where the position of the interaction is 1.
def df_to_sparse(df_pos: pd.DataFrame, user_to_idx, item_to_idx) -> csr_matrix:
    rows = df_pos['user_id'].astype(str).map(user_to_idx).values
    cols = df_pos['spotify_id'].astype(str).map(item_to_idx).values
    data = np.ones(len(df_pos), dtype=np.float32)
    mat = coo_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx))).tocsr()
    return mat

#For each user, k interacted items are randomly selected as the test set, 
# and the rest are used as the training matrix.
def leave_k_out_split(mat: csr_matrix, k: int = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    mat = mat.tocsr()
    n_users, _ = mat.shape
    train_rows, train_cols, train_data = [], [], []
    test_items = defaultdict(list)

    for u in range(n_users):
        start, end = mat.indptr[u], mat.indptr[u+1]
        items = mat.indices[start:end]
        if len(items) == 0:
            continue
        k_eff = min(k, len(items))
        test_idx = rng.choice(items, size=k_eff, replace=False)
        test_set = set(test_idx.tolist())
        for it in items:
            if it in test_set:
                test_items[u].append(it)
            else:
                train_rows.append(u); train_cols.append(it); train_data.append(1.0)

    train_mat = coo_matrix((train_data, (train_rows, train_cols)), shape=mat.shape).tocsr()
    return train_mat, test_items

#The similarity sparsity is achieved by pruning each row of a sparse matrix to retain only 
# the k largest values ​​in that row (Top-K).
def _topk_sparse_rows(mat: csr_matrix, k: int) -> csr_matrix:
    mat = mat.tocsr().astype(np.float32)
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    new_indptr = [0]
    new_indices = []
    new_data = []
    for r in range(mat.shape[0]):
        start, end = indptr[r], indptr[r+1]
        row_idx = indices[start:end]
        row_val = data[start:end]
        if len(row_val) > k:
            sel = np.argpartition(row_val, -k)[-k:]
            order = sel[np.argsort(row_val[sel])[::-1]]
            new_indices.extend(row_idx[order].tolist())
            new_data.extend(row_val[order].tolist())
        else:
            new_indices.extend(row_idx.tolist()); new_data.extend(row_val.tolist())
        new_indptr.append(len(new_indices))
    return csr_matrix((np.array(new_data, dtype=np.float32), np.array(new_indices), np.array(new_indptr)), shape=mat.shape)

#Item-based collaborative filtering (Item-CF) generates Top-K recommendations for each user.
def recommend_itemcf(train_mat: csr_matrix, k_sim: int = 50, topk: int = 10,
                     sig_weight: bool = False, sig_cap: int = 50):

    item_sim = cosine_similarity(train_mat.T, dense_output=False)

    item_sim = _topk_sparse_rows(item_sim, k=k_sim)

    if sig_weight:
        bool_ui = train_mat.astype(bool).astype(np.float32).tocsr()
        overlap = (bool_ui.T @ bool_ui).tocsr()
        item_sim = item_sim.tolil(); overlap = overlap.tolil()
        for i in range(item_sim.shape[0]):
            cols = item_sim.rows[i]
            if not cols: continue
            ov_vals = []
            for j in cols:
                if j in overlap.rows[i]:
                    pos = overlap.rows[i].index(j); ov = overlap.data[i][pos]
                else:
                    ov = 0.0
                w = min(ov, sig_cap) / float(sig_cap) if sig_cap > 0 else 1.0
                ov_vals.append(w)
            item_sim.data[i] = (np.array(item_sim.data[i]) * np.array(ov_vals)).tolist()
        item_sim = item_sim.tocsr()

    seen = train_mat.tocsr()
    scores = seen @ item_sim

    scores[seen.nonzero()] = 0.0

    recs = []
    for u in range(scores.shape[0]):
        row = scores.getrow(u)
        if row.nnz == 0: recs.append([]); continue
        top_items = row.toarray().ravel().argsort()[::-1][:topk]
        recs.append(top_items.tolist())
    return recs

#Latent semantic recommendation is performed using NMF (Non-negative Matrix Factorization). 
# For each user, the top k item indices are retrieved in descending order of score.
def recommend_nmf(train_mat: csr_matrix, n_components: int = 64, topk: int = 10, max_iter: int = 200):
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=max_iter)
    W = nmf.fit_transform(train_mat)
    H = nmf.components_
    scores = W @ H
    seen = train_mat.tocsr()
    scores[seen.nonzero()] = -np.inf
    recs = np.argsort(-scores, axis=1)[:, :topk]
    return [row.tolist() for row in recs]



def hit_rate_at_k(recs, test_items, k: int = 10) -> float:
    users, hits = 0, 0
    for u, items in test_items.items():
        users += 1
        topk = set(recs[u][:k]) if recs[u] else set()
        if any(x in topk for x in items):
            hits += 1
    return hits / max(users, 1)


def ndcg_at_k(recs, test_items, k: int = 10) -> float:
    def dcg(rel_scores):
        return sum(rel / np.log2(i+2) for i, rel in enumerate(rel_scores))
    ndcgs = []
    for u, items in test_items.items():
        if not recs[u]: ndcgs.append(0.0); continue
        top = recs[u][:k]
        rel = [1.0 if it in items else 0.0 for it in top]
        ideal = sorted(rel, reverse=True)
        denom = dcg(ideal) or 1.0
        ndcgs.append(dcg(rel) / denom)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def compute_metrics_curve(make_recs_fn, test_items, k_max: int):
    # Generate recommendations once at k_max, then slice
    recs_full = make_recs_fn(topk=k_max)
    ks = list(range(1, k_max+1))
    hrs, ndcgs = [], []
    for k in ks:
        hrs.append(hit_rate_at_k(recs_full, test_items, k=k))
        ndcgs.append(ndcg_at_k(recs_full, test_items, k=k))
    return ks, hrs, ndcgs


def plot_metrics(ks, hrs, ndcgs, out_path: Path):
    plt.figure()
    plt.plot(ks, hrs, label="HitRate@K")
    plt.plot(ks, ndcgs, label="NDCG@K")
    plt.xlabel("K"); plt.ylabel("Score"); plt.title("HitRate & NDCG vs K")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_hist_counts(full_mat: csr_matrix, out_user: Path, out_item: Path, bins: int = 50):
    # EDA histograms on FULL dataset (pre-split).
    # per-user interactions
    user_counts = np.diff(full_mat.indptr)
    plt.figure(); plt.hist(user_counts, bins=bins)
    plt.xlabel("Interactions per user"); plt.ylabel("Frequency"); plt.title("User Interactions (Full Dataset)")
    plt.tight_layout(); plt.savefig(out_user); plt.close()

    # per-item popularity
    full_mat_csc = full_mat.tocsc()
    item_counts = np.diff(full_mat_csc.indptr)
    plt.figure(); plt.hist(item_counts, bins=bins)
    plt.xlabel("Interactions per item"); plt.ylabel("Frequency"); plt.title("Item Popularity (Full Dataset)")
    plt.tight_layout(); plt.savefig(out_item); plt.close()


def build_popularity(train_mat: csr_matrix) -> np.ndarray:
    return np.asarray(train_mat.sum(axis=0)).ravel()


def recommend_with_popularity(popularity: np.ndarray, seen_row, topk: int) -> List[int]:
    scores = popularity.copy()
    if seen_row.nnz > 0:
        scores[seen_row.indices] = -np.inf
    return np.argsort(-scores)[:topk].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='./data/processed_music_data_no_negatives.csv')
    ap.add_argument('--assume_liked_if_missing', action='store_true')
    ap.add_argument('--model', type=str, choices=['nmf','itemcf'], default='nmf')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--leave_k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--nmf_components', type=int, default=64)
    ap.add_argument('--nmf_max_iter', type=int, default=200)
    ap.add_argument('--itemcf_neighbors', type=int, default=50)
    ap.add_argument('--sig_weight', action='store_true')
    ap.add_argument('--sig_cap', type=int, default=50)
    ap.add_argument('--export_recs', type=str, default=None)
    ap.add_argument('--plot_dir', type=str, default=None, help='If set, save figures to this directory')
    args = ap.parse_args()

    # Load & map
    df_pos = load_interactions(args.data, assume_liked_if_missing=args.assume_liked_if_missing)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_mappings(df_pos)
    full_mat = df_to_sparse(df_pos, user_to_idx, item_to_idx)

    # EDA plots on FULL dataset
    if args.plot_dir:
        outdir = Path(args.plot_dir); outdir.mkdir(parents=True, exist_ok=True)
        plot_hist_counts(full_mat, outdir / "user_interactions_hist.png", outdir / "item_popularity_hist.png")

    # Train/test split
    train_mat, test_items = leave_k_out_split(full_mat, k=args.leave_k, seed=args.seed)


    popularity = build_popularity(train_mat)

    def make_recs(topk: int):
        if args.model == 'itemcf':
            recs_local = recommend_itemcf(train_mat, k_sim=args.itemcf_neighbors, topk=topk,
                                          sig_weight=args.sig_weight, sig_cap=args.sig_cap)
        else:
            recs_local = recommend_nmf(train_mat, n_components=args.nmf_components, topk=topk, max_iter=args.nmf_max_iter)
        # Fallback for empty rows
        for u in range(train_mat.shape[0]):
            if not recs_local[u]:
                recs_local[u] = recommend_with_popularity(popularity, train_mat.getrow(u), topk=topk)
        return recs_local

    # Evaluate at K
    recs = make_recs(args.k)
    hr = hit_rate_at_k(recs, test_items, k=args.k)
    ndcg = ndcg_at_k(recs, test_items, k=args.k)
    print("\n=== Evaluation ===")
    print(f"Model: {args.model}")
    print(f"HitRate@{args.k}:   {hr:.4f}")
    print(f"NDCG@{args.k}:      {ndcg:.4f}")

    if args.plot_dir:
        ks, hrs, ndcgs = compute_metrics_curve(make_recs, test_items, k_max=args.k)
        plot_metrics(ks, hrs, ndcgs, outdir / "metrics_vs_k.png")
        print(f"[Plot] Saved figures to {outdir}")

    if args.export_recs:
        rows = []
        for u_idx, items in enumerate(recs):
            user = idx_to_user.get(u_idx, str(u_idx))
            for rank, it_idx in enumerate(items[:args.k], start=1):
                rows.append((user, idx_to_item.get(it_idx, str(it_idx)), rank))
        out = pd.DataFrame(rows, columns=['user_id', 'spotify_id', 'rank'])
        out.to_csv(args.export_recs, index=False)
        print(f"[Export] Saved recommendations to {args.export_recs}")


if __name__ == "__main__":
    main()
