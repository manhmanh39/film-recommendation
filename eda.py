import os
import subprocess
from zipfile import ZipFile
import pandas as pd
from IPython.display import display
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# == Download & extract dataset == #

ds_url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
temp_dir = "/tmp"

subprocess.run(["wget", "-P", temp_dir, ds_url])

with ZipFile(os.path.join(temp_dir, "ml-32m.zip")) as z_obj:
    z_obj.extractall(path=temp_dir)

# == Load movies and ratings into dataframes == #

movies_path = os.path.join(temp_dir, "ml-32m", "movies.csv")
ratings_path = os.path.join(temp_dir, "ml-32m", "ratings.csv")
tags_path = os.path.join(temp_dir, "ml-32m", "tags.csv")
links_path = os.path.join(temp_dir, "ml-32m", "links.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

display(movies)
display(ratings)
display(tags)
display(links)

# == Create transactions for each user contain movie id in time order == #

top_k = 500

transactions = (
    ratings.sort_values("timestamp")
    .groupby("userId")["movieId"]
    .apply(list)
    .tolist()
)

movies_popularity = ratings["movieId"].value_counts()
display(movies_popularity)

top_movies = movies_popularity.head(top_k).index

transactions = [[m for m in t if m in top_movies] for t in transactions]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

print("\nOne-hot encoded data preview:")
print(df_transactions.head())

min_sup = 0.1
min_confidence = 0.5

num_transactions = len(transactions)

frequent_itemsets = fpgrowth(
    df_transactions,
    min_support=min_sup,
    use_colnames=True,
    max_len=4,
    verbose=1,
)

rules = association_rules(
    frequent_itemsets,
    num_itemsets=num_transactions,
    metric="confidence",
    min_threshold=0.3,
)

print("\n===== FP-Growth Results =====")
print(frequent_itemsets.sort_values("support"))

print("\nFP-Growth Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

work_dir = os.getcwd()

frequent_itemsets.to_csv(os.path.join(work_dir, "frequent_itemsets.csv"))
rules.to_csv(os.path.join(work_dir, "rules.csv"))


# === Scatter plot === #

plt.figure(figsize=(8, 6))

rules_filtered = rules[(rules["confidence"] > 0.7) & (rules["lift"] > 1.5)]

plt.scatter(
    rules_filtered["support"],
    rules_filtered["confidence"],
    s=rules_filtered["lift"] * 50,
    alpha=0.6,
    color="tab:orange",
)

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("FP-Growth: Support vs Confidence (size = lift)")

plt.show()
