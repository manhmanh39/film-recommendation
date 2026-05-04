from model import BERT4Rec
from dataset import MovieLenDataset
import pandas as pd
import os
import subprocess
from zipfile import ZipFile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ==  Variables == #

batch_size = 32
num_epochs = 50
val_iter = 1
mask_rate = 0.2
max_len = 200
min_len = 5
d_model = 128
n_heads = 4
n_layers = 4
dropout = 0.2
lr = 5e-5
top_k = 200

model_name = "bert4rec"

base_dir = "../data"
experiment_dir = f"{model_name}_{d_model}"
if not os.path.isdir(os.path.join(base_dir, experiment_dir)):
    os.mkdir(os.path.join(base_dir, experiment_dir))

checkpoint_path = os.path.join(base_dir, experiment_dir, "checkpoint.pt")
losses_path = os.path.join(base_dir, experiment_dir, "losses.csv")
validation_path = os.path.join(base_dir, experiment_dir, "validation.csv")

ds_url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
temp_dir = "/tmp"

extracted_path = os.path.join(temp_dir, "ml-32m")

if not os.path.exists(extracted_path):
    print("Dữ liệu chưa có, đang tải...")
    subprocess.run(["wget", "-P", temp_dir, ds_url])
    with ZipFile(os.path.join(temp_dir, "ml-32m.zip")) as z_obj:
        z_obj.extractall(path=temp_dir)
else:
    print("Đã tìm thấy dữ liệu tại /tmp/ml-32m, bỏ qua bước tải.")

movies_path = os.path.join(temp_dir, "ml-32m", "movies.csv")
ratings_path = os.path.join(temp_dir, "ml-32m", "ratings.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# == Initialize datasets with Caching == #
# Dùng chung cache với MetaBERT4Rec — cùng bộ data split để so sánh công bằng
cache_file = os.path.join(temp_dir, "dataset_32m_cache.pkl")

if os.path.exists(cache_file):
    print("--- Đang nạp Dataset từ bộ nhớ đệm (Cache)... ---")
    with open(cache_file, "rb") as f:
        train_ds, popularity_val_ds, random_val_ds, trending_val_ds = pickle.load(f)
    print("--- Nạp thành công! Chuẩn bị vào vòng lặp training. ---")
else:
    print("--- Không tìm thấy cache. Bắt đầu khởi tạo Dataset (Sẽ tốn thời gian)... ---")

    train_ds = MovieLenDataset(
        movies=movies,
        ratings=ratings,
        max_len=max_len,
        min_len=min_len,
        strides=100,
        split="train",
    )

    popularity_val_ds = MovieLenDataset(
        movies=movies,
        ratings=ratings,
        max_len=max_len,
        min_len=min_len,
        split="val",
        top_k=top_k,
        negative_rule="popularity",
    )

    random_val_ds = MovieLenDataset(
        movies=movies,
        ratings=ratings,
        max_len=max_len,
        min_len=min_len,
        split="val",
        top_k=top_k,
        negative_rule="random",
    )

    trending_val_ds = MovieLenDataset(
        movies=movies,
        ratings=ratings,
        max_len=max_len,
        min_len=min_len,
        split="val",
        top_k=top_k,
        negative_rule="trending",
    )

    print("--- Đang lưu Dataset vào cache cho lần sau... ---")
    with open(cache_file, "wb") as f:
        pickle.dump((train_ds, popularity_val_ds, random_val_ds, trending_val_ds), f)
    print("--- Đã lưu xong! ---")

# == Initialize DataLoaders == #
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

popularity_val_loader = DataLoader(
    dataset=popularity_val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

random_val_loader = DataLoader(
    dataset=random_val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

trending_val_loader = DataLoader(
    dataset=trending_val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

# == Load checkpoint == #

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print(f"Resuming from epoch {checkpoint['epoch']}, start_epoch = {checkpoint['epoch'] + 1}")
else:
    checkpoint = None
    print("No checkpoint found, starting from epoch 1")

# == Model == #

model = BERT4Rec(
    max_len=max_len,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    vocab_size=len(movies) + 2,
    dropout=dropout,
)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if not module.weight.requires_grad:
            return
        nn.init.trunc_normal_(module.weight, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


model.apply(init_weights)

if checkpoint is not None:
    model.load_state_dict(checkpoint["model"])

model.to(device)

# == Training tools == #

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

# == CSV == #
losses_file_exists = os.path.exists(losses_path)

# == Training script == #


def validate_one_epoch(model, val_loader, device, val_type, epoch, K_list=[1, 5, 10]):
    model.eval()
    metrics = {f"{metric}@{k}": 0.0 for metric in ["Recall", "NDCG", "MRR"] for k in K_list}
    metrics.update({"MRR": 0.0, "MeanRank": 0.0})

    total_samples = 0
    st = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating {val_type}"):
            idx = batch["input"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)
            # Trong leave-one-out, item mục tiêu thường là item cuối cùng của chuỗi label
            target_items = batch["label"][:, -1].to(device) 

            # 🔥 FULL RANKING: Đặt candidates=None
            # BERT4Rec sẽ trả về logits cho toàn bộ vocab: [B, T, Vocab_size]
            logits = model.forward(
                idx=idx,
                key_padding_mask=key_padding_mask,
                candidates=None
            )

            # Lấy logits tại vị trí cuối cùng (vị trí [mask] hoặc vị trí dự báo)
            logits = logits[:, -1, :] # Shape: [B, Vocab_size]

            # Tính Rank của phim đúng trong toàn bộ Vocab
            target_scores = logits.gather(1, target_items.unsqueeze(1)) 
            # Đếm số lượng phim có điểm cao hơn phim đúng
            ranks = (logits > target_scores).sum(dim=1) + 1 
            ranks_float = ranks.float()

            for K in K_list:
                hit = (ranks <= K).float()
                metrics[f"Recall@{K}"] += hit.sum().item()
                
                # NDCG chuẩn cho 1 item mục tiêu (IDCG = 1)
                ndcg = torch.where(
                    hit > 0,
                    1.0 / torch.log2(ranks_float + 1),
                    torch.zeros_like(hit)
                )
                metrics[f"NDCG@{K}"] += ndcg.sum().item()
                
                mrr_k = torch.where(
                    ranks <= K,
                    1.0 / ranks_float,
                    torch.zeros_like(ranks_float)
                )
                metrics[f"MRR@{K}"] += mrr_k.sum().item()

            metrics["MRR"] += (1.0 / ranks_float).sum().item()
            metrics["MeanRank"] += ranks_float.sum().item()
            total_samples += idx.size(0)

    for key in metrics:
        metrics[key] /= total_samples

    et = time.perf_counter()

    row = {
        "epoch": epoch,
        "val_type": val_type,
        "sec_per_batch": (et - st) / total_samples,
        **metrics,
    }

    pd.DataFrame([row]).to_csv(
        validation_path,
        mode='a',
        header=not os.path.exists(validation_path),
        index=False,
    )

    return row


scaler = torch.amp.GradScaler('cuda')


def train_one_epoch(model, batch, accumulation_steps):
    model.train()

    idx = batch["input"].to(device)
    label = batch["label"].to(device)
    token_mask = batch["token_mask"].to(device)
    key_padding_mask = batch["key_padding_mask"].to(device)

    with torch.amp.autocast('cuda'):
        logits = model.forward(idx=idx, key_padding_mask=key_padding_mask)

        flatten_token_mask = torch.flatten(token_mask)
        V = logits.shape[2]
        y_pred = logits.view(-1, V)[flatten_token_mask]
        y_true = torch.flatten(label)[flatten_token_mask]

        loss = criterion(y_pred, y_true) / accumulation_steps

    scaler.scale(loss).backward()

    return loss.item() * accumulation_steps


# == Early Stopping == #
patience = 7
best_ndcg = 0.0 if checkpoint is None else checkpoint.get("ndcg", 0.0)
counter = 0 if checkpoint is None else checkpoint.get("es_counter", 0)
early_stop = False
best_model_path = os.path.join(base_dir, experiment_dir, "best_model.pt")

start_epoch = 1 if checkpoint is None else checkpoint["epoch"] + 1
accumulation_steps = 4
optimizer.zero_grad()

for epoch in range(start_epoch, num_epochs + 1):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    epoch_loss_sum = 0.0

    for step, batch in pbar:
        loss = train_one_epoch(model, batch, accumulation_steps)
        epoch_loss_sum += loss

        pd.DataFrame([{"epoch": epoch, "step": step, "loss": loss}]).to_csv(
            losses_path,
            mode='a',
            header=not losses_file_exists,
            index=False,
        )
        losses_file_exists = True

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        pbar.set_description(desc=f"Loss: {loss:.4f}")

    scheduler.step()

    avg_loss = epoch_loss_sum / len(train_loader)
    print(f"\n[Epoch {epoch}/{num_epochs}] Average loss: {avg_loss:.4f}")

    if epoch % val_iter == 0:
        epoch_ndcg = 0.0

        for val_loader, val_type in [
            (popularity_val_loader, "popularity"),
            (random_val_loader, "random"),
            (trending_val_loader, "trending"),
        ]:
            row = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                val_type=val_type,
                device=device,
                epoch=epoch,
            )
            epoch_ndcg += row["NDCG@10"]
            print(f"  [{val_type}] NDCG@10: {row['NDCG@10']:.4f}")

        current_ndcg = epoch_ndcg / 3
        print(f"Validation result for Epoch {epoch}: avg NDCG@10 = {current_ndcg:.4f}")

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "ndcg": best_ndcg}, best_model_path)
            print(f"==> NEW BEST MODEL! Saved at epoch {epoch} with NDCG@10: {best_ndcg:.4f}")
        else:
            counter += 1
            print(f"==> EarlyStopping counter: {counter} out of {patience}")

        if counter >= patience:
            print(f"!!! [STOP] Early stopping triggered at epoch {epoch} !!!")
            early_stop = True

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ndcg": best_ndcg,
        "es_counter": counter,
    }, checkpoint_path)

    if early_stop:
        break