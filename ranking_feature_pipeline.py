import pprint

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

from typing import Optional, List, Tuple, Dict, Text
from collections import defaultdict

anime_features = ("title","popularity","episodes","format","year","studio","source","avg_score") #list(anime_df.columns)

def _create_feature_dict() -> Dict[Text, List[tf.Tensor]]:
    """Helper function for creating an empty feature dict for defaultdict."""
    return {"title": [], "user_rating": [], "user":[]}

def _sample_list(
        feature_lists: Dict[Text, List[tf.Tensor]],
        num_examples_per_list: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function for sampling a list example from given feature lists."""
    if random_state is None:
        random_state = np.random.Generator()

    sampled_indices = random_state.choice(
        range(len(feature_lists["title"])),
        size=num_examples_per_list,
        replace=False,
    )
    sampled_ratings = [
        feature_lists["user_rating"][idx]
        for idx in sampled_indices
    ]
    #sampled_title = [
    #    feature_lists["title"][idx]
    #    for idx in sampled_indices
    #]

    return (
        #tf.stack(sampled_title, 0),
        {anime_feature: tf.stack([feature_lists["title"][idx][anime_feature] for idx in sampled_indices],0) for anime_feature in anime_features},
        #tf.stack([feature_lists["title"][idx] for idx in sampled_indices],0),
        tf.stack(sampled_ratings, 0),
    )

def sample_listwise(
    rating_dataset: tf.data.Dataset,
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Function for converting the MovieLens 100K dataset to a listwise dataset.
    Args:
        rating_dataset:
            The MovieLens ratings dataset loaded from TFDS with features
            "movie_title", "user_id", and "user_rating".
        num_list_per_user:
            An integer representing the number of lists that should be sampled for
            each user in the training dataset.
        num_examples_per_list:
            An integer representing the number of movies to be sampled for each list
            from the list of movies rated by the user.
        seed:
            An integer for creating `np.random.RandomState`.
    Returns:
        A tf.data.Dataset containing list examples.
        Each example contains three keys: "user_id", "movie_title", and
        "user_rating". "user_id" maps to a string tensor that represents the user
        id for the example. "movie_title" maps to a tensor of shape
        [sum(num_example_per_list)] with dtype tf.string. It represents the list
        of candidate movie ids. "user_rating" maps to a tensor of shape
        [sum(num_example_per_list)] with dtype tf.float32. It represents the
        rating of each movie in the candidate list.
    """
    example_lists_by_user = defaultdict(_create_feature_dict)
    for example in rating_dataset:
        user_id = example["user_id"].numpy()
        example_lists_by_user[user_id]["user"].append(
            example["mean_score"])
        example_lists_by_user[user_id]["user"].append(
            example["score_count"])
        example_lists_by_user[user_id]["user"].append(
            example["completed_count"])
        example_lists_by_user[user_id]["title"].append(
            example)
        example_lists_by_user[user_id]["user_rating"].append(
            example["user_rating"])

    tensor_slices = {"user_id": [], "mean_score": [], "score_count":[], "completed_count":[], **{feature: [] for feature in anime_features}, "user_rating": []}
    #tensor_slices = {"user_id": [], "anime":[], "user_rating": [], "mean_score": [], "score_count":[], "completed_count":[]}
    for user_id, feature_lists in example_lists_by_user.items():
        for _ in range(num_list_per_user):

        # Drop the user if they don't have enough ratings.
            if len(feature_lists["title"]) < num_examples_per_list:
                continue

            sampled_anime, sampled_rating = _sample_list(
                feature_lists,
                num_examples_per_list,
            )
            tensor_slices["user_id"].append(user_id)
            tensor_slices["mean_score"].append(feature_lists["user"][0])
            tensor_slices["score_count"].append(feature_lists["user"][1])
            tensor_slices["completed_count"].append(feature_lists["user"][2])

            _ = [tensor_slices[feature].append(sampled_anime[feature]) for feature in anime_features]
            tensor_slices["user_rating"].append(sampled_rating)
            #tensor_slices["anime"].append(sampled_anime)

    return tf.data.Dataset.from_tensor_slices(tensor_slices)

shuffled = ratings.shuffle(2_000_000, reshuffle_each_iteration=False)

# We sample 50 lists for each user for the training data. For each list we
# sample 5 movies from the movies the user rated.
train = shuffled.take(320_000)
test = shuffled.skip(320_000).take(80_000)


train = sample_listwise(
    train,
    num_list_per_user=5,
    num_examples_per_list=10,
)
test = sample_listwise(
    test,
    num_list_per_user=1,
    num_examples_per_list=10,
)
