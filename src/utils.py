import os
import re
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MovieLenDataset

def init_weights(module):
    """Khởi tạo trọng số chuẩn cho Transformer."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if not module.weight.requires_grad:
            return
        nn.init.trunc_normal_(module.weight, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

def prepare_dataloaders(data_dir="../data", max_len=200, min_len=5, batch_size=32, val_batch_size=16):
    # ĐỔI TÊN CACHE ĐỂ KHÔNG DÙNG LẠI DATA CŨ
    cache_file = os.path.join(data_dir, "dataset_timesplit_cache.pkl") 
    
    extracted_path = os.path.join(data_dir, "ml-32m")
    movies_path = os.path.join(extracted_path, "movies.csv")
    ratings_path = os.path.join(extracted_path, "ratings.csv")
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(f"❌ Không tìm thấy dữ liệu tại {extracted_path}.")

    if os.path.exists(cache_file):
        print("📦 Đang nạp Dataset SẠCH (Time-Split) từ Cache...")
        with open(cache_file, "rb") as f:
            train_ds, val_ds, vocab_size = pickle.load(f)
    else:
        print("🧹 Đang thanh lọc và CẮT DỮ LIỆU THEO THỜI GIAN...")
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        
        def extract_year(title):
            match = re.search(r'\((\d{4})\)', str(title))
            return int(match.group(1)) if match else 2000
        movies['year'] = movies['title'].apply(extract_year)
        
        # 1. Lọc nhiễu cơ bản
        movie_counts = ratings['movieId'].value_counts()
        ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= 50].index)]
        user_std = ratings.groupby('userId')['rating'].std()
        ratings = ratings[ratings['userId'].isin(user_std[user_std > 0].index)]
        
        # 2. CHIA TÁCH DỮ LIỆU THEO THỜI GIAN (01/01/2021)
        SPLIT_DATE = 1609459200
        train_ratings = ratings[ratings['timestamp'] < SPLIT_DATE]
        test_ratings = ratings[ratings['timestamp'] >= SPLIT_DATE]
        
        # Lưu tập Test ra file riêng để chạy Simulation A/B Test chiều nay
        test_ratings_path = os.path.join(data_dir, "test_ratings_ab.csv")
        test_ratings.to_csv(test_ratings_path, index=False)
        print(f"✅ Đã lưu tập Simulation (Tương lai): {len(test_ratings)} dòng ra {test_ratings_path}")
        
        vocab_size = len(movies) + 2
        
        print(f"⏳ Đang xây dựng Dataset trên tập Train/Val trong quá khứ ({len(train_ratings)} dòng)...")
        # CHỈ TRUYỀN `train_ratings` VÀO ĐỂ TRAIN
        train_ds = MovieLenDataset(movies=movies, ratings=train_ratings, max_len=max_len, min_len=min_len, strides=50, split="train")
        val_ds = MovieLenDataset(movies=movies, ratings=train_ratings, max_len=max_len, min_len=min_len, strides=50, split="val")
        
        with open(cache_file, "wb") as f:
            pickle.dump((train_ds, val_ds, vocab_size), f)
            
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, vocab_size

def train_epoch(model, loader, criterion, optimizer, scaler, accum_steps, device, is_meta=False):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc="🔥 Training")
    
    for step, batch in enumerate(pbar):
        idx = batch["input"].to(device)
        label = batch["label"].to(device)
        token_mask = batch["token_mask"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)
        
        with torch.amp.autocast('cuda'):
            if is_meta:
                genres = batch["genres"].to(device)
                logits = model(idx, genres, key_padding_mask=key_padding_mask)
            else:
                logits = model(idx, key_padding_mask=key_padding_mask)
            
            flatten_mask = torch.flatten(token_mask)
            if flatten_mask.sum() == 0: 
                continue
            
            V = logits.shape[-1]
            y_pred = logits.view(-1, V)[flatten_mask]
            y_true = torch.flatten(label)[flatten_mask]
            
            loss = criterion(y_pred, y_true) / accum_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        epoch_loss += loss.item() * accum_steps
        pbar.set_postfix({"Loss": f"{loss.item() * accum_steps:.4f}"})
        
    return epoch_loss / len(loader)


def validate_epoch(model, loader, val_type, device, is_meta=False, K_list=[1, 5, 10, 20]):
    model.eval()
    
    metrics = {f"Recall@{K}": 0.0 for K in K_list}
    metrics.update({f"NDCG@{K}": 0.0 for K in K_list})
    metrics.update({f"MRR@{K}": 0.0 for K in K_list})
    metrics["MRR"] = 0.0
    
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"🔎 Validation [{val_type}]"):
            idx = batch["input"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)
            
            # Lấy 101 candidates từ dataset
            candidates = batch["candidates"].to(device)
            
            with torch.amp.autocast('cuda'):
                if is_meta:
                    genres = batch["genres"].to(device)
                    # Ép model chỉ sinh logits cho 101 candidates này
                    logits = model(idx, genres, key_padding_mask=key_padding_mask, candidates=candidates)
                else:
                    logits = model(idx, key_padding_mask=key_padding_mask, candidates=candidates)
                    
            # Target (item thật) nằm ở vị trí cuối cùng trong mảng candidates
            target_scores = logits[:, -1].unsqueeze(1)
            ranks = (logits > target_scores).sum(dim=1) + 1

            for K in K_list:
                hit = (ranks <= K).float()
                metrics[f"Recall@{K}"] += hit.sum().item()
                metrics[f"NDCG@{K}"] += (hit / torch.log2(ranks.float() + 1)).sum().item()
                metrics[f"MRR@{K}"] += (hit / ranks.float()).sum().item()
            
            metrics["MRR"] += (1.0 / ranks.float()).sum().item()
            total_samples += idx.size(0)
            
    for k in metrics:
        metrics[k] /= total_samples
        
    return metrics, metrics.get("NDCG@10", 0.0)