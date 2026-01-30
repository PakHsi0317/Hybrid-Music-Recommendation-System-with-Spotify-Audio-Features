import math
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_id_maps(df: pd.DataFrame, user_col: str, item_col: str):
    """Map raw user/item IDs to contiguous indices [0..n-1]."""
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()
    user2idx = {u:i for i,u in enumerate(unique_users)}
    item2idx = {it:i for i,it in enumerate(unique_items)}
    return user2idx, item2idx

def encode_ids(df: pd.DataFrame, user_col: str, item_col: str, user2idx, item2idx):
    """Add encoded id columns u_idx and i_idx."""
    u_idx = df[user_col].map(user2idx).values.astype(np.int64)
    i_idx = df[item_col].map(item2idx).values.astype(np.int64)
    return u_idx, i_idx

def leave_k_out_split(df: pd.DataFrame, user_col: str, k_val: int = 1, k_test: int = 1, seed: int = 42):
    assert k_val >= 0 and k_test >= 0
    rng = np.random.RandomState(seed)

    train_idx_all, val_idx_all, test_idx_all = [], [], []
    for _, grp in df.groupby(user_col):
        idxs = grp.index.values.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        if n >= k_val + k_test + 1:
            test_idx = idxs[:k_test]
            val_idx = idxs[k_test:k_test + k_val]
            train_idx = idxs[k_test + k_val:]
            test_idx_all.extend(list(test_idx))
            val_idx_all.extend(list(val_idx))
            train_idx_all.extend(list(train_idx))
        elif n >= (k_val + 1):
            # enough for val, but not test
            val_idx = idxs[:k_val]
            train_idx = idxs[k_val:]
            val_idx_all.extend(list(val_idx))
            train_idx_all.extend(list(train_idx))
        else:
            # too few, all to train
            train_idx_all.extend(list(idxs))

    return (
        df.loc[train_idx_all].reset_index(drop=True),
        df.loc[val_idx_all].reset_index(drop=True),
        df.loc[test_idx_all].reset_index(drop=True),
    )

class MBCFDataset(Dataset):
    def __init__(self, u_idx, i_idx, n_items: int, user_pos_items: dict, with_neg=False):
        self.u = torch.as_tensor(u_idx, dtype=torch.long)
        self.i = torch.as_tensor(i_idx, dtype=torch.long)
        self.n_items = n_items
        self.user_pos = user_pos_items
        self.with_neg = with_neg

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = int(self.u[idx].item())
        i = int(self.i[idx].item())

        if not self.with_neg:
            return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
        )
        
        pos_set = self.user_pos.get(u, set())
        j = np.random.randint(0, self.n_items)
        tries = 0
        while j in pos_set and tries < 20:
            j = np.random.randint(0, self.n_items)
            tries += 1
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
        )
    
class MBCF(nn.Module):
    def __init__(self, n_users, n_items, dim=64, init_std=0.01):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, dim)
        self.item_factors = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
         
        nn.init.normal_(self.user_factors.weight, std=init_std)
        nn.init.normal_(self.item_factors.weight, std=init_std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        # score (logit) = dot + biases
        dot = (self.user_factors(u) * self.item_factors(i)).sum(dim=-1, keepdim=True)
        logits = dot + self.user_bias(u) + self.item_bias(i) + self.global_bias
        return logits.squeeze(-1)
    
def dcg_at_k(rank: int):
    if rank <= 0:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def ndcg_at_k(ranks, k: int) -> float:
    vals = [ dcg_at_k(int(r)) if int(r) <= k else 0.0 for r in ranks ]
    return float(np.mean(vals)) if len(vals) else 0.0

def hit_rate_at_k(ranks, k: int) -> float:
    ranks = np.asarray(ranks)
    return float(np.mean(ranks <= k)) if len(ranks) else 0.0

def MRR(ranks):
    ranks = np.asarray(ranks)
    return float(np.mean(ranks)) if len(ranks) else 0.0

@torch.no_grad()
def recommend_for_user(
    model: MBCF, 
    user_raw_id, 
    user2idx, 
    item2idx, 
    user_pos_items, 
    device, 
    k: int = 10
):
    if user_raw_id not in user2idx:
        raise ValueError("Unknown user_id: {}".format(user_raw_id))
    u = user2idx[user_raw_id]
    n_items = len(item2idx)
    # Build mask of items already interacted/liked in train
    seen = user_pos_items.get(u, set())

    # Score all items
    all_items = torch.arange(n_items, dtype=torch.long, device=device)
    user_vec = torch.full((n_items,), u, dtype=torch.long, device=device)
    logits = model(user_vec, all_items)
    scores = torch.sigmoid(logits)

    # Mask seen
    mask = torch.ones(n_items, dtype=torch.bool, device=device)
    for it in seen:
        mask[it] = False
    masked_scores = scores.clone()
    masked_scores[~mask] = -1.0  # push seen ones down

    topk_scores, topk_idx = torch.topk(masked_scores, k)
    # invert item2idx
    idx2item = {v:k for k,v in item2idx.items()}
    recs = [(idx2item[int(i.item())], float(s.item())) for i,s in zip(topk_idx, topk_scores)]
    return recs

@torch.no_grad()
def evaluate_ranking(
    model: MBCF,
    eval_df: pd.DataFrame,
    user_pos_items: dict,
    n_items: int,
    device,
    k_list=(10,),
    all_items: bool = False,
    num_negatives: int = 999,
    user_col: str = 'u',
    item_col: str = 'i',
):
    rng = np.random.RandomState(123)
    ranks = []

    eval_by_user = {}
    for _, row in eval_df.iterrows():
        eval_by_user.setdefault(int(row[user_col]), []).append(int(row[item_col]))

    for u, pos_items in eval_by_user.items():
        seen = user_pos_items.get(u, set())
        if all_items:
            cand = np.arange(n_items, dtype=np.int64)
            mask = np.ones(n_items, dtype=bool)
            for it in seen:
                mask[it] = False
            for it in pos_items:
                mask[it] = True
            cand = cand[mask]
        else:
            cand = set()
            tries = 0
            while len(cand) < num_negatives and tries < num_negatives * 20:
                j = int(rng.randint(0, n_items))
                if (j not in seen) and (j not in pos_items):
                    cand.add(j)
                tries += 1
            cand = np.fromiter(cand, dtype=np.int64)
            cand = np.concatenate([cand, np.array(pos_items, dtype=np.int64)], axis=0)

        u_vec = torch.full((len(cand),), u, dtype=torch.long, device=device)
        items_t = torch.as_tensor(cand, dtype=torch.long, device=device)
        scores = torch.sigmoid(model(u_vec, items_t)).cpu().numpy()

        for p in pos_items:
            pos_idx = np.where(cand == p)[0]
            s_pos = scores[pos_idx[0]]
            rank = 1 + int(np.sum(scores > s_pos))
            ranks.append(rank)

    results = {}
    for K in k_list:
        results[f'HR@{K}'] = hit_rate_at_k(ranks, K)
        results[f'NDCG@{K}'] = ndcg_at_k(ranks, K)

    results[f'MRR'] = MRR(ranks)
    return results

if __name__ == "__main__":
    seed = 42
    device = torch.device('cuda')

    csv_path = "data/processed_music_data_no_negatives.csv"
    user_col = "user_id"
    item_col = "spotify_id"

    k_val = 1
    k_test = 1
    with_neg = True

    model_dim = 64

    epochs = 100
    batch_size = 4096
    lr = 1e-3
    weight_decay = 1e-6

    num_negatives = 999

    logging.basicConfig(filename='MBCF_MMF.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"csv_path: {csv_path}")
    logger.info(f"user_col: {user_col}")
    logger.info(f"item_col: {item_col}")
    logger.info(f"k_val: {k_val}")
    logger.info(f"k_test: {k_test}")
    logger.info(f"with_neg: {with_neg}")
    logger.info(f"model_dim: {model_dim}")
    logger.info(f"epochs: {epochs}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"lr: {lr}")
    logger.info(f"weight_decay: {weight_decay}")
    logger.info(f"num_negatives: {num_negatives}")

    df = pd.read_csv(csv_path)
    print(f"read CSV: {csv_path}")

    set_seed(seed)

    user2idx, item2idx = build_id_maps(df, user_col, item_col)
    u_idx_all, i_idx_all = encode_ids(df, user_col, item_col, user2idx, item2idx)

    train_df, val_df, test_df = leave_k_out_split(pd.DataFrame({'u': u_idx_all, 'i': i_idx_all}), 'u', k_val, k_test, seed)
    print('Split train, val and test')
    
    n_users = len(user2idx)
    n_items = len(item2idx)

    user_pos_items = {u: set() for u in range(n_users)}
    for _, row in train_df.iterrows():
        user_pos_items[int(row['u'])].add(int(row['i']))
    print('Build per-user positive sets')

    # Datasets
    train_ds = MBCFDataset(train_df['u'].values, train_df['i'].values, n_items, user_pos_items, with_neg)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    print('Create training dataloader')

    # Model
    model = MBCF(n_users, n_items, model_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    vali_result = evaluate_ranking(model, val_df, user_pos_items, n_items, device, k_list=(5,10,20),
                                            all_items=False, num_negatives=num_negatives)
    logger.info(f"Vali: {vali_result}")
    test_result = evaluate_ranking(model, test_df, user_pos_items, n_items, device, k_list=(5,10,20),
                                            all_items=False, num_negatives=num_negatives)
    logger.info(f"Test: {test_result}", )

    print('Start training')
    for ep in range(1, epochs+1):
        model.train()
        total_loss, total_n = 0.0, 0

        for u, i_pos, j_neg in train_loader:
            u = u.to(device)
            i_pos = i_pos.to(device)
            pos_logits = model(u, i_pos)
            loss = bce(pos_logits, torch.ones_like(pos_logits))

            if with_neg:
                j_neg = j_neg.to(device)
                neg_logits = model(u, j_neg)
                loss += bce(neg_logits, torch.zeros_like(neg_logits))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(u)
            total_n += len(u)

        tr_loss = total_loss / max(1, total_n)
        logger.info(f"Epoch {ep:02d} | training BCE {tr_loss:.4f}")

        if ep % 5 == 0:
            vali_result = evaluate_ranking(model, val_df, user_pos_items, n_items, device, k_list=(5,10,20),
                                            all_items=False, num_negatives=num_negatives)
            logger.info(f"Vali: {vali_result}")
            test_result = evaluate_ranking(model, test_df, user_pos_items, n_items, device, k_list=(5,10,20),
                                            all_items=False, num_negatives=num_negatives)
            logger.info(f"Test: {test_result}", )

    # logger.info(f"finish training, test on full dataset")

    # test_result = evaluate_ranking(model, test_df, user_pos_items, n_items, device, k_list=(5,10,20),
    #                                 all_items=(ep == epochs), num_negatives=num_negatives)
    # logger.info(f"Test: {test_result}", )
    # sample_user = df[user_col].iloc[0]
    # recs = recommend_for_user(model, sample_user, user2idx, item2idx, user_pos_items, device, k=10)
    # print(f"\nTop-10 recommendations for user {sample_user}:")
    # for item_id, score in recs:
    #     print(f"{item_id}\t{score:.4f}")

    logger.info(f"Done\n\n")

