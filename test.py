import os
import time
import subprocess
from zipfile import ZipFile
import pandas as pd
from IPython.display import display

patterns = pd.read_csv("/content/drive/MyDrive/frequent_itemsets.csv")
rules = pd.read_csv("/content/drive/MyDrive/rules.csv")

patterns["itemsets"] = patterns["itemsets"].apply(eval)
rules["antecedents"] = rules["antecedents"].apply(eval)
rules["consequents"] = rules["consequents"].apply(eval)

display(patterns)
display(rules)

ds_url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
temp_dir = "/tmp"

subprocess.run(["wget", "-P", temp_dir, ds_url])

with ZipFile(os.path.join(temp_dir, "ml-32m.zip")) as z_obj:
    z_obj.extractall(path=temp_dir)

movies_path = os.path.join(temp_dir, "ml-32m", "movies.csv")
ratings_path = os.path.join(temp_dir, "ml-32m", "ratings.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

display(movies)
display(ratings)

transactions = (
    ratings.sort_values("timestamp")
    .groupby("userId")["movieId"]
    .apply(list)
    .tolist()
)

movie2idx = {id: idx + 1 for idx, id in enumerate(movies["movieId"].unique())}

itemsets = (
    rules[(rules["confidence"] > 0.8) & (rules["antecedents"].str.len() > 1)]
    .groupby("antecedents")
    .count()
    .reset_index()["antecedents"]
)
display(itemsets)

matches = [p for p in itemsets if p.issubset(transactions[0])]
print(len(matches))

st = time.perf_counter()
for i in range(128):
    transaction = transactions[i]
    matches = [p for p in itemsets if p.issubset(transactions[i])]
et = time.perf_counter()
print("Elasped time: ", et - st)
