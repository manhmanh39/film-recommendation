import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import MetaBERT4Rec
from utils import prepare_dataloaders, train_epoch, validate_epoch, init_weights

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. CONFIGS TỐI ƯU (CHỐNG OOM) ──────────────────────────────────────────
d_model = 64
n_heads = 2
n_layers = 2
batch_size, val_batch_size = 32, 256
num_epochs, val_iter, patience = 50, 1, 5
accum_steps = 8

experiment_dir = f"../data/metabert4rec_{d_model}_optimized"
os.makedirs(experiment_dir, exist_ok=True)
checkpoint_path = os.path.join(experiment_dir, "checkpoint.pt")
losses_path = os.path.join(experiment_dir, "losses.csv")
validation_metrics_path = os.path.join(experiment_dir, "validation_metrics.csv")

# ─── 2. DATALOADERS ─────────────────────────────────────────────────────────
train_loader, val_loader, vocab_size = prepare_dataloaders(
    data_dir="../data", max_len=200, min_len=5, 
    batch_size=batch_size, val_batch_size=val_batch_size
)

# ─── 3. INITIALIZE ──────────────────────────────────────────────────────────
model = MetaBERT4Rec(max_len=200, num_genres=20, d_model=d_model, n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01) # Thêm weight decay
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.amp.GradScaler('cuda')

# ─── 4. RESUME ──────────────────────────────────────────────────────────────
start_epoch, best_ndcg, es_counter = 1, 0.0, 0
if os.path.exists(checkpoint_path):
    print("🔄 Phục hồi từ Checkpoint...")
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch, best_ndcg, es_counter = ckpt["epoch"] + 1, ckpt.get("ndcg", 0.0), ckpt.get("es_counter", 0)

# ─── 5. TRAINING LOOP ───────────────────────────────────────────────────────
for epoch in range(start_epoch, num_epochs + 1):
    avg_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, accum_steps, device, is_meta=True)
    scheduler.step()
    
    pd.DataFrame([{"epoch": epoch, "loss": avg_loss}]).to_csv(losses_path, mode='a', header=not os.path.exists(losses_path), index=False)
    
    if epoch % val_iter == 0:
        metrics, ndcg = validate_epoch(model, val_loader, "Validation", device, is_meta=True)
        row = {"epoch": epoch, **metrics}
        pd.DataFrame([row]).to_csv(validation_metrics_path, mode='a', header=not os.path.exists(validation_metrics_path), index=False)
        
        print(f"🏆 Epoch {epoch} | Loss: {avg_loss:.4f} | NDCG@10: {ndcg:.4f}")
        
        if ndcg > best_ndcg:
            best_ndcg, es_counter = ndcg, 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "ndcg": best_ndcg}, os.path.join(experiment_dir, "best_model.pt"))
            print("⭐ Đã lưu Best Model mới!")
        else:
            es_counter += 1
            print(f"⚠️ Early Stopping: {es_counter}/{patience}")
            if es_counter >= patience: break
            
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "ndcg": best_ndcg, "es_counter": es_counter}, checkpoint_path)