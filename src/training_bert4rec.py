from model import BERT4Rec, MetaBERT4Rec
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ==  Variables == #

batch_size = 64
num_epochs = 50
val_iter = 5
mask_rate = 0.2
max_len = 200
min_len = 5
d_model = 64
n_heads = 2
n_layers = 2
dropout = 0.2
lr = 1e-5
top_k = 200

model_name = "bert4rec"

base_dir = ""
experiment_dir = f"{model_name}_{d_model}"
if not os.path.isdir(os.path.join(base_dir, experiment_dir)):
    os.mkdir(os.path.join(base_dir, experiment_dir))

checkpoint_path = os.path.join(base_dir, experiment_dir, "checkpoint.pt")
losses_path = os.path.join(base_dir, experiment_dir, "losses.csv")
validation_path = os.path.join(base_dir, experiment_dir, "validation.csv")

ds_url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
temp_dir = "/tmp"

# == Download ml-20m dataset == #

subprocess.run(["wget", "-P", temp_dir, ds_url])

with ZipFile(os.path.join(temp_dir, "ml-20m.zip")) as z_obj:
    z_obj.extractall(path=temp_dir)

movies_path = os.path.join(temp_dir, "ml-20m", "movies.csv")
ratings_path = os.path.join(temp_dir, "ml-20m", "ratings.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# == Initialize datasets == #

train_ds = MovieLenDataset(
    movies=movies,
    ratings=ratings,
    max_len=max_len,
    min_len=min_len,
    strides=20,
    split="train",
)

popularity_val_ds = MovieLenDataset(
    movies=movies,
    ratings=ratings,
    max_len=max_len,
    min_len=min_len,
    top_k=top_k,
    split="val",
    negative_rule="popularity",
)

random_val_ds = MovieLenDataset(
    movies=movies,
    ratings=ratings,
    max_len=max_len,
    min_len=min_len,
    top_k=top_k,
    split="val",
    negative_rule="random",
)

trending_val_ds = MovieLenDataset(
    movies=movies,
    ratings=ratings,
    max_len=max_len,
    min_len=min_len,
    top_k=top_k,
    split="val",
    negative_rule="trending",
)

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
else:
    checkpoint = None

# == Model == #

model = BERT4Rec(
    max_len=max_len,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    vocab_size=len(movies) + 2,
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
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=lr,
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=num_epochs,
)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

# == losses and validation dataframe == #

if os.path.exists(losses_path):
    losses_df = pd.read_csv(losses_path)
else:
    losses_df = pd.DataFrame(
        columns=[
            "epoch",
            "step",
            "loss",
        ]
    )

if os.path.exists(validation_path):
    validation_df = pd.read_csv(validation_path)
else:
    columns = [
        "epoch",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@1",
        "MRR@5",
        "MRR@10",
        "MRR",
        "NDCG@1",
        "NDCG@5",
        "NDCG@10",
        "MeanRank",
    ]
    validation_df = pd.DataFrame(columns=columns)

# == Training script == #


def validate_one_epoch(
    model,
    val_loader,
    device,
    validation_df,
    val_type,
    epoch,
    K_list=[1, 5, 10],
):
    model.eval()

    # Accumulators
    metrics = {
        f"{metric}@{k}": 0.0
        for metric in ["Recall", "NDCG", "MRR"]
        for k in K_list
    }

    # Global metrics
    metrics["MRR"] = 0.0
    metrics["MeanRank"] = 0.0

    total_samples = 0
    st = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(val_loader):
            idx = batch["input"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)
            candidates = batch["candidates"].to(device)  # [B, C]

            # Forward
            logits = model.forward(
                idx=idx,
                key_padding_mask=key_padding_mask,
                candidates=candidates,
            )  # [B, C]

            B, C = logits.shape
            target_idx = C - 1  # always last position

            # Sort logits
            sorted_indices = torch.argsort(logits, dim=1, descending=True)

            # Find rank of target
            target_positions = (sorted_indices == target_idx).nonzero(
                as_tuple=False
            )

            ranks = torch.zeros(B, device=device, dtype=torch.long)
            ranks[target_positions[:, 0]] = (
                target_positions[:, 1] + 1
            )  # 1-indexed

            ranks_float = ranks.float()

            # === Metrics ===
            for K in K_list:
                hit = (ranks <= K).float()

                # Recall@K
                metrics[f"Recall@{K}"] += hit.sum().item()

                # NDCG@K
                ndcg = torch.where(
                    hit > 0,
                    1.0 / torch.log2(ranks_float + 1),
                    torch.zeros_like(hit),
                )
                metrics[f"NDCG@{K}"] += ndcg.sum().item()

                # MRR@K
                mrr_k = torch.where(
                    ranks <= K,
                    1.0 / ranks_float,
                    torch.zeros_like(ranks_float),
                )
                metrics[f"MRR@{K}"] += mrr_k.sum().item()

            # === Global MRR ===
            metrics["MRR"] += (1.0 / ranks_float).sum().item()

            # === Mean Rank ===
            metrics["MeanRank"] += ranks_float.sum().item()

            total_samples += B

    # Average
    for key in metrics:
        metrics[key] /= total_samples

    et = time.perf_counter()
    total_run_time = et - st

    # Append
    row = {
        "epoch": epoch,
        "val_type": val_type,
        "sec_per_batch": total_run_time / total_samples,
        **metrics,
    }
    validation_df.loc[len(validation_df)] = row

    return validation_df


def train_one_epoch(model, optimizer, batch):
    model.train()

    idx = batch["input"].to(device)
    label = batch["label"].to(device)
    token_mask = batch["token_mask"].to(device)
    key_padding_mask = batch["key_padding_mask"].to(device)

    logits = model.forward(
        idx=idx,
        key_padding_mask=key_padding_mask,
    )

    flatten_token_mask = torch.flatten(token_mask)
    V = logits.shape[2]
    y_pred = logits.view(-1, V)[flatten_token_mask]
    y_true = torch.flatten(label)[flatten_token_mask]

    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


start_epoch = 1 if checkpoint is None else checkpoint["epoch"] + 1
for epoch in range(start_epoch, num_epochs + 1):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in pbar:
        loss = train_one_epoch(model, optimizer, batch)
        losses_df.loc[len(losses_df)] = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
        }

        pbar.set_description(desc=f"Loss: {loss}")

    scheduler.step()

    epoch_loss = losses_df[losses_df["epoch"] == epoch]["loss"].mean()
    print(f"{epoch}/{num_epochs}: Average loss: {epoch_loss}")

    if epoch % val_iter == 0:
        validation_df = validate_one_epoch(
            model=model,
            val_loader=popularity_val_loader,
            val_type="popularity",
            device=device,
            validation_df=validation_df,
            epoch=epoch,
        )
        validation_df = validate_one_epoch(
            model=model,
            val_loader=random_val_loader,
            val_type="random",
            device=device,
            validation_df=validation_df,
            epoch=epoch,
        )
        validation_df = validate_one_epoch(
            model=model,
            val_loader=trending_val_loader,
            val_type="trending",
            device=device,
            validation_df=validation_df,
            epoch=epoch,
        )
        validation_df.to_csv(validation_path)

        print("Validation result")
        print(validation_df[validation_df["epoch"] == epoch])

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    losses_df.to_csv(losses_path)
