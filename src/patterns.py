import pandas as pd
import numpy as np
from IPython.display import display


class PatternsPerUser:
    def __init__(self, movies, ratings, patterns, rules):
        self.movies = movies
        self.ratings = ratings
        self.patterns = patterns
        self.rules = self.rules

        id2idx = {id: idx + 1 for idx, id in enumerate(self.movies["movieId"])}
        self.ratings["movie_idx"] = self.ratings["movieId"].map(id2idx)
        self.movies["movie_idx"] = self.movies["movieId"].map(id2idx)

        self.patterns[""]


if __name__ == "__main__":
    patterns = pd.read_csv("/content/drive/MyDrive/frequent_itemsets.csv")
    rules = pd.read_csv("/content/drive/MyDrive/rules.csv")

    patterns["itemsets"] = patterns["itemsets"].apply(eval).apply(list)
    rules["antecedents"] = rules["antecedents"].apply(eval).apply(list)
    rules["consequents"] = rules["consequents"].apply(eval).apply(list)

    display(patterns)
    display(rules)
