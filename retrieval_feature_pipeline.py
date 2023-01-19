import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

#!curl -L -o './dataset_sv/train/dataset.arrow' 'https://drive.google.com/uc?id=10YL_6Aeu_fqOxcF02-4uEbtSv3WHc9y0&confirm=t'

anime_ratings = pd.read_csv("dataset/processed_anime_ratings.csv")
anime_df = pd.read_csv("dataset/processed_anime.csv")

# Select the basic features.
ratings = anime_ratings[["user_id","score","title_romaji","mean_score","score_count","completed_count","episodes","format","source","year","popularity","avg_score","is_adult","studio","status","is_fav","repeat"]].rename(columns={
    "title_romaji": "title",
    "score": "user_rating"
    })

ratings = tf.data.Dataset.from_tensor_slices((dict(ratings[["title","episodes","format","year","source","popularity","avg_score","is_adult","studio","user_id","mean_score","score_count","completed_count","user_rating"]])))
anime = tf.data.Dataset.from_tensor_slices((dict(anime_df)))

shuffled = ratings.shuffle(2_000_000, reshuffle_each_iteration=False)
train = shuffled.take(1_750_000)
validation = shuffled.skip(1_750_000).take(50_000)
test = shuffled.skip(1_800_000).take(200_000)

unique_anime_titles = np.unique(anime_ratings["title_romaji"])
unique_user_ids = np.unique(anime_ratings["user_id"])

unique_features = {
    "episodes" : np.unique(anime_ratings["episodes"]),
    "popularity" : np.unique(anime_ratings["popularity"]),
    "format" : np.unique(anime_ratings["format"]),
    "year" : np.unique(anime_ratings["year"]),
    "studio" : np.unique(anime_ratings["studio"]),
    "source" : np.unique(anime_ratings["source"]),
    "avg_score" : np.unique(anime_ratings["avg_score"]),
    "status" : np.unique(anime_ratings["status"]),
    "mean_score" : np.unique(anime_ratings["mean_score"]),
    "score_count" : np.unique(anime_ratings["score_count"]),
    "completed_count" : np.unique(anime_ratings["completed_count"]),
}

class AnimeModel(tf.keras.Model):

    def __init__(self, embedding_dims: int = 64, features: tuple = ()):

        super().__init__()

        max_tokens = 50_000

        self.embedding_dims = embedding_dims
        self.feature_embeddings = {}

        self.title_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_anime_titles,mask_token=None),
        tf.keras.layers.Embedding(len(unique_anime_titles) + 1, self.embedding_dims)
        ])

        for feature in features:
            if feature in ("format","studio","source"):
                self.feature_embeddings[feature] = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_features[feature], mask_token=None),
                tf.keras.layers.Embedding(len(unique_features[feature]) + 1, self.embedding_dims),
            ])
            else:
                self.feature_embeddings[feature] = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_features[feature], mask_token=None),
                tf.keras.layers.Embedding(len(unique_features[feature]) + 1, self.embedding_dims),
            ])

    def call(self, inputs) -> tf.Tensor:
        return tf.concat([self.title_embedding(inputs['title'])] +
                        [self.feature_embeddings[k](inputs[k]) for k in self.feature_embeddings],
                            axis=1)


class UserModel(tf.keras.Model):

    def __init__(self, embedding_dims: int = 64, features: tuple = ()):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.feature_embeddings = {}

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, self.embedding_dims),
        ])

        for feature in features:
            if feature in ("format"):
                self.feature_embeddings[feature] = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_features[feature], mask_token=None),
                tf.keras.layers.Embedding(len(unique_features[feature]) + 1, self.embedding_dims),
            ])
            else:
                self.feature_embeddings[feature] = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_features[feature], mask_token=None),
                tf.keras.layers.Embedding(len(unique_features[feature]) + 1, self.embedding_dims),
            ])

    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat([self.user_embedding(inputs['user_id'])] +
                        [self.feature_embeddings[k](inputs[k]) for k in self.feature_embeddings],
                            axis=1)

# t_user_features=("mean_score","score_count","completed_count")
# t_anime_features=("popularity","episodes","format","year","studio","source","avg_score")
# anime_model = AnimeModel(64, t_anime_features)
# user_model = UserModel(64, t_user_features)
# cand_embeds = ratings.batch(512).map(anime_model)
# query_embeds = ratings.batch(512).map(user_model)
# anime_embeds = anime.batch(128).map(anime_model)

# proc_ratings = tf.data.Dataset.zip((cand_embeds, query_embeds))
# anime_embeds.save('proc_anime', compression="GZIP")
# proc_ratings.save('proc_ratings', compression="GZIP")
