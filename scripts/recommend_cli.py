import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.model import BERT4Rec, MetaBERT4Rec, SASRec
from src.dataset import GENRES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend top-K movies from a list of watched titles."
    )
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--model", default="sasrec", choices=["bert4rec", "metabert4rec", "sasrec"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--titles", default=None, help="Semicolon-separated list of movie titles.")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke test without checkpoints.")
    return parser.parse_args()


def resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def load_movies(data_dir):
    movies_path = os.path.join(data_dir, "ml-32m", "movies.csv")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Missing movies.csv at: {movies_path}")
    return pd.read_csv(movies_path)


def normalize_title(title):
    return " ".join(str(title).strip().lower().split())


def build_title_index(movies):
    title_to_id = {}
    for _, row in movies.iterrows():
        title_to_id[normalize_title(row["title"])] = int(row["movieId"])
    return title_to_id


def resolve_titles_to_ids(titles, title_to_id, movies):
    movie_ids = []
    for raw_title in titles:
        key = normalize_title(raw_title)
        if key in title_to_id:
            movie_ids.append(title_to_id[key])
            continue

        candidates = movies[movies["title"].str.lower().str.contains(key)]
        if candidates.empty:
            raise ValueError(f"Title not found: {raw_title}")

        sample = candidates["title"].head(5).tolist()
        raise ValueError(
            f"Title not found: {raw_title}. Did you mean one of: {sample}?"
        )
    return movie_ids


def build_id_maps(movies):
    movie_ids = movies["movieId"].tolist()
    id2idx = {mid: idx + 1 for idx, mid in enumerate(movie_ids)}
    idx2id = {idx + 1: mid for idx, mid in enumerate(movie_ids)}
    return id2idx, idx2id


def build_genre_map(movies):
    genres_map = {}
    for _, row in movies.iterrows():
        genre_vec = [0.0] * len(GENRES)
        if isinstance(row["genres"], str):
            for g in row["genres"].split("|"):
                if g in GENRES:
                    genre_vec[GENRES.index(g)] = 1.0
        genres_map[int(row["movieId"])] = genre_vec
    return genres_map


def build_sequence(movie_ids, id2idx, max_len, vocab_size):
    seq_idx = [id2idx[m] for m in movie_ids if m in id2idx]
    if len(seq_idx) == 0:
        raise ValueError("No valid movie IDs found in the input list.")

    max_hist = max_len - 1
    seq_idx = seq_idx[-max_hist:]
    mask_token = vocab_size - 1
    seq_idx = seq_idx + [mask_token]

    pad_len = max_len - len(seq_idx)
    if pad_len > 0:
        seq_idx = [0] * pad_len + seq_idx

    idx_tensor = torch.tensor([seq_idx], dtype=torch.long)
    key_padding_mask = idx_tensor == 0
    return idx_tensor, key_padding_mask, mask_token


def build_genre_tensor(movie_ids, genres_map, max_len):
    max_hist = max_len - 1
    history = movie_ids[-max_hist:]
    seq_genres = [genres_map.get(mid, [0.0] * len(GENRES)) for mid in history]
    seq_genres.append([0.0] * len(GENRES))

    pad_len = max_len - len(seq_genres)
    if pad_len > 0:
        seq_genres = [[0.0] * len(GENRES)] * pad_len + seq_genres
    return torch.tensor([seq_genres], dtype=torch.float32)


def find_checkpoint(data_dir, model_name):
    matches = []
    for root, _, files in os.walk(data_dir):
        if "best_model.pt" not in files:
            continue
        folder_name = os.path.basename(root).lower()
        if model_name in folder_name:
            matches.append(os.path.join(root, "best_model.pt"))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple checkpoints found for {model_name}: {matches}. Use --ckpt to choose one."
        )
    return None


def default_ckpt_path(model_name, d_model):
    if model_name == "sasrec":
        return f"../data/sasrec_ce_{d_model}/best_model.pt"
    if model_name == "bert4rec":
        return f"../data/bert4rec_{d_model}/best_model.pt"
    if model_name == "metabert4rec":
        return f"../data/metabert4rec_{d_model}/best_model.pt"
    raise ValueError(f"Unknown model name: {model_name}")


def build_model(model_name, vocab_size, args, device):
    if model_name == "sasrec":
        return SASRec(args.max_len, args.d_model, args.n_heads, args.n_layers, vocab_size).to(device)
    if model_name == "bert4rec":
        return BERT4Rec(args.max_len, args.d_model, args.n_heads, args.n_layers, vocab_size).to(device)
    if model_name == "metabert4rec":
        return MetaBERT4Rec(args.max_len, len(GENRES), args.d_model, args.n_heads, args.n_layers, vocab_size).to(device)
    raise ValueError(f"Unknown model name: {model_name}")


def load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])


def pick_default_max_len(model_name):
    if model_name == "sasrec":
        return 100
    return 200


def run_smoke():
    print("[SMOKE] recommend_cli.py is ready.")


def main():
    args = parse_args()
    if args.smoke:
        run_smoke()
        return

    if args.max_len is None:
        args.max_len = pick_default_max_len(args.model)

    data_dir = resolve_path(SCRIPT_DIR, args.data_dir)
    movies = load_movies(data_dir)
    title_to_id = build_title_index(movies)
    id2idx, idx2id = build_id_maps(movies)
    genres_map = build_genre_map(movies)
    vocab_size = len(movies) + 2

    if args.titles is None:
        raw = input("Enter movie titles separated by ';': ").strip()
        titles = [t.strip() for t in raw.split(args.delimiter) if t.strip()]
    else:
        titles = [t.strip() for t in args.titles.split(args.delimiter) if t.strip()]

    if len(titles) < 10:
        raise ValueError("Please provide at least 10 movie titles.")

    movie_ids = resolve_titles_to_ids(titles, title_to_id, movies)
    idx_tensor, key_padding_mask, mask_token = build_sequence(
        movie_ids, id2idx, args.max_len, vocab_size
    )
    genre_tensor = build_genre_tensor(movie_ids, genres_map, args.max_len)

    device = get_device()
    model = build_model(args.model, vocab_size, args, device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = find_checkpoint(data_dir, args.model)
    if ckpt_path is None:
        ckpt_path = default_ckpt_path(args.model, args.d_model)
    ckpt_path = resolve_path(SCRIPT_DIR, ckpt_path)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    idx_tensor = idx_tensor.to(device)
    key_padding_mask = key_padding_mask.to(device)
    genre_tensor = genre_tensor.to(device)

    with torch.no_grad():
        with get_autocast(device):
            if args.model == "metabert4rec":
                logits = model(idx_tensor, genre_tensor, key_padding_mask=key_padding_mask, candidates=None)
            else:
                logits = model(idx_tensor, key_padding_mask=key_padding_mask, candidates=None)

    logits = logits[0]
    for mid in movie_ids:
        if mid in id2idx:
            logits[id2idx[mid]] = -float("inf")
    logits[0] = -float("inf")
    logits[mask_token] = -float("inf")

    scores, top_indices = torch.topk(logits, args.topk)

    print("\n=== Watched Movies ===")
    for title in titles:
        print(f"- {title}")

    print("\n=== Recommendations ===")
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), scores.tolist()), start=1):
        movie_id = idx2id.get(idx, None)
        if movie_id is None:
            continue
        row = movies[movies["movieId"] == movie_id].iloc[0]
        print(f"{rank:2d}. [{score:6.2f}] {row['title']}")


if __name__ == "__main__":
    main()
