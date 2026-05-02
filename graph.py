import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import numpy as np

plt.rcParams["figure.dpi"] = 300


# == Compare loss == #

d_model = 64
stride = 1000

bert4rec_loss_path = f"data/bert4rec_{d_model}/losses.csv"
metabert4rec_loss_path = f"data/metabert4rec_{d_model}/losses.csv"

bert4rec_loss = pd.read_csv(bert4rec_loss_path)
metabert4rec_loss = pd.read_csv(metabert4rec_loss_path)

x = np.linspace(1, 50, len(bert4rec_loss["loss"][::stride]))

plt.plot(
    x,
    bert4rec_loss["loss"][::stride],
    color="tab:blue",
    label="BERT4Rec",
)
plt.plot(
    x,
    metabert4rec_loss["loss"][::stride],
    color="tab:orange",
    label="MetaBERT4Rec",
)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(f"report/fig/losses_compare_d_model_{d_model}.png")
plt.show()

# == Compare validation result == #

val_results = {}


for d_model in [32, 64, 128]:
    metabert4rec_val_path = f"data/metabert4rec_{d_model}/validation.csv"
    metabert4rec_val = pd.read_csv(metabert4rec_val_path)

    result = metabert4rec_val[-3:]
    popularity_result = result[result.index % 3 == 0].iloc[0]
    trending_result = result[result.index % 3 == 2].iloc[0]
    random_result = result[result.index % 3 == 1].iloc[0]

    val_results[d_model] = {
        "popularity": popularity_result,
        "trending": trending_result,
        "random": random_result,
    }

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fontsize = 16

for i, metric in enumerate(["Recall@5", "MRR@5", "NDCG@5"]):
    dimensions = [32, 64, 128]

    popularity_perf = [
        val_results[d_model]["popularity"][metric] for d_model in dimensions
    ]
    trending_perf = [
        val_results[d_model]["trending"][metric] for d_model in dimensions
    ]
    random_perf = [
        val_results[d_model]["random"][metric] for d_model in dimensions
    ]

    axes[i].plot(
        [1, 2, 3],
        popularity_perf,
        marker=".",
        label="Popularity",
    )
    axes[i].plot(
        [1, 2, 3],
        trending_perf,
        marker=".",
        label="Trending",
    )
    axes[i].plot(
        [1, 2, 3],
        random_perf,
        marker=".",
        label="Random",
    )
    axes[i].set_xticks([1, 2, 3], dimensions)
    axes[i].set_ylabel(metric, fontsize=fontsize)
    axes[i].set_xlabel("Dimensionality", fontsize=fontsize)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.2),
    fontsize=fontsize,
)

plt.show()

plt.tight_layout()
plt.savefig("report/fig/negative_rule_comparison.png")
