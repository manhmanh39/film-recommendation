import os
import subprocess
from zipfile import ZipFile
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

WEEK_IN_SEC = 604800
DAY_IN_SEC = 86400

GENRES = [
    "(no genres listed)", 
    "Action",
    "Adventure",
    "Animation",
    "Children",      
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",          
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def get_genre_matrix(movies_df):
    """Vectorized genre encoding using Pandas dummies"""
    dummies = movies_df["genres"].str.get_dummies(sep="|")
    return dummies.reindex(columns=GENRES, fill_value=0).values


def generate_mask(seq, mask_rate):
    """
    Randomly generate a mask for the given sequence. The mask rate specify how much of the sequence is masked
    True value indicate the position will be masked.
    """
    return torch.rand(len(seq)) < mask_rate


def parse_week(ratings):
    """
    Parse the week where the current rating is on.
    ratings where the timestamp is less than 1 day away from the start of a week will be parsed as previous week
    """
    return np.where(
        (ratings["timestamp"] % WEEK_IN_SEC) > DAY_IN_SEC,
        ratings["timestamp"] // WEEK_IN_SEC,
        (ratings["timestamp"] // WEEK_IN_SEC) - 1,
    )


class MovieLenDataset(Dataset):
    """
    Args:
        movies: the movies dataframe
        ratings: the ratings dataframe
        negative_rule: the rule used to determine how negative items are sampled (popularity|trending|random)
        top_k: the k movies will be used for negative sample
        min_len: the minimum user history length to be used, otherwise that user will be removed.
        max_len: the maximum user history length to be used, otherwise that user will be removed.
        mask_rate: the proportion of the sequence to be masked randomly
        split: the target split the dataset is used for (train|val|test)
    """

    def __init__(
        self,
        movies,
        ratings,
        min_len=5,
        max_len=200,
        negative_rule="popularity",
        strides=1,
        mask_rate=0.2,
        top_k=100,
        split="train",
    ):
        super().__init__()

        self.split = split
        self.negative_rule = negative_rule
        self.max_len = max_len
        self.mask_rate = mask_rate
        self.top_k = top_k
        self.negative_samples = []

        self._prepare(movies, ratings)
        self._build_sequences(min_len, strides)
        self.MASK_ID = len(self.movies) + 1

        if self.split == "train":
            return

        if self.negative_rule == "popularity":
            # ✅ SỬA Ở ĐÂY: Lấy cả index (movie_idx) và values (tần suất) để làm weights
            movie_counts = self.ratings["movie_idx"].value_counts()
            
            for i in tqdm(range(len(self.seqs)), desc="Negative Sampling (Popularity)"):
                seq = self.seqs[i]["seq"]
                
                # Lọc ra những phim user CHƯA xem trong chuỗi hiện tại
                valid_movies = movie_counts[~movie_counts.index.isin(seq)]
                
                # Lấy mẫu ngẫu nhiên 100 phim, xác suất rớt vào tỷ lệ thuận với độ phổ biến (weights)
                sample = valid_movies.sample(
                    n=self.top_k, 
                    weights=valid_movies.values, 
                    replace=False
                ).index.to_list()
                
                self.negative_samples.append(sample)
        elif self.negative_rule == "trending":
            movies_by_trending = (
                self.ratings.groupby(["movie_idx", "week"])["movieId"]
                .agg("count")
                .to_frame("count")
                .reset_index()
                .sort_values(["week", "count"], ascending=False)
            )

            for i in tqdm(range(len(self.seqs))):
                seq = self.seqs[i]["seq"]
                week = self.seqs[i]["week"]
                sample = (
                    movies_by_trending[movies_by_trending["week"] == week]
                    .head(self.top_k)["movie_idx"]
                    .to_list()
                )
                self.negative_samples.append(sample)
        elif self.negative_rule == "random":
            for i in tqdm(range(len(self.seqs))):
                seq = self.seqs[i]["seq"]
                sample = (
                    self.movies[~self.movies["movie_idx"].isin(seq)][
                        "movie_idx"
                    ]
                    .sample(self.top_k)
                    .to_list()
                )
                self.negative_samples.append(sample)

    def _prepare(self, movies, ratings):
        ratings["week"] = parse_week(ratings)
        id2idx = {id: idx + 1 for idx, id in enumerate(movies["movieId"])}
        ratings["movie_idx"] = ratings["movieId"].map(id2idx)
        movies["movie_idx"] = movies["movieId"].map(id2idx)
        self.genres_lookup = np.vstack(
            [np.zeros(len(GENRES)), get_genre_matrix(movies)]
        )
        self.movies = movies
        self.ratings = ratings

    def _build_sequences(self, min_len, strides):
        grouped = self.ratings.sort_values("timestamp").groupby("userId")
        user_data = grouped.agg({"movie_idx": list, "week": list})

        iterator = tqdm(
            user_data.iterrows(),
            total=len(user_data),
            desc=f"Initialize dataset for {self.split}",
        )

        seqs = []
        for _, row in iterator:
            hist, weeks = row["movie_idx"], row["week"]
            if len(hist) < min_len:
                continue

            if self.split == "train":
                for i in range(
                    0, max(len(hist) - self.max_len - 2, 1), strides
                ):
                    seq = hist[i : i + self.max_len]
                    seqs.append({"seq": seq})

            elif self.split == "val" or self.split == "test":
                offset = 1 if self.split == "val" else 0
                idx_end = len(hist) - offset
                seq = hist[max(idx_end - self.max_len, 0) : idx_end]
                target_week = weeks[-1]
                seqs.append({"seq": seq, "week": target_week})

        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]["seq"]
        genres = self.genres_lookup[seq]
        seq = torch.tensor(seq)
        genres = torch.from_numpy(genres).long()
        pad = (max(0, self.max_len - len(seq)), 0)
        padded_seq = F.pad(seq, pad, value=0)
        padded_genres = F.pad(genres, (0, 0, pad[0], pad[1]))
        key_padding_mask = padded_seq == 0

        if self.split == "train":
            token_mask = generate_mask(seq, self.mask_rate)
            padded_token_mask = F.pad(token_mask, pad, value=False)
            label = padded_seq.clone()
            padded_seq[padded_token_mask] = self.MASK_ID
            padded_genres[padded_token_mask] = 0

            return {
                "input": padded_seq,
                "label": label,
                "genres": padded_genres,
                "token_mask": padded_token_mask,
                "key_padding_mask": key_padding_mask,
            }
            
        elif self.split == "val" or self.split == "test":
            negatives = torch.tensor(self.negative_samples[idx])
            negatives_pad = (max(0, self.top_k - len(negatives)), 0)
            padded_negatives = F.pad(negatives, negatives_pad)
            token_mask = torch.tensor([False] * (len(seq) - 1) + [True])
            padded_token_mask = F.pad(token_mask, pad, value=False)
            label = padded_seq.clone()

            padded_seq[padded_token_mask] = self.MASK_ID
            target = seq[-1]
            padded_genres[padded_token_mask] = 0

            return {
                "input": padded_seq,
                "label": label,
                "genres": padded_genres,
                "token_mask": padded_token_mask,
                "key_padding_mask": key_padding_mask,
                "candidates": torch.cat(
                    (padded_negatives, target.unsqueeze(0))
                ),
            }
