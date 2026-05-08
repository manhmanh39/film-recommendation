# Film Recommendation Offline A/B Simulation

This repo includes an offline A/B simulation that compares two models on a small group of validation samples. It is intended for demo or course projects where online traffic is not available.

## What the script does

- Samples a small set of validation sequences (treated as "users")
- Splits them into two groups (A/B)
- Computes Hit@K and NDCG@K using the target item in the validation set
- Reports bootstrap confidence intervals for the difference

## Run

From the repo root:

```bash
python scripts/sim_ab_offline.py --model-a bert4rec --model-b metabert4rec --n-users 12 --topk 10
```

## CLI demo (friends enter watched titles)

This demo takes a list of movie titles and returns top-K recommendations.

```bash
python scripts/recommend_cli.py --model sasrec --titles "Toy Story (1995);Jumanji (1995);Grumpier Old Men (1995)"
```

If you want to type titles interactively, omit `--titles` and the script will prompt you.

Optional: run a smoke test without checkpoints or data:

```bash
python scripts/sim_ab_offline.py --smoke
```

## Notes

- The script expects model checkpoints in `../data/*/best_model.pt` unless you pass `--ckpt-a` and `--ckpt-b`.
- The validation samples come from the same preprocessing as training (via `prepare_dataloaders`).
- Small sample size is suitable for demonstration, not for statistical proof.
