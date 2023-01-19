import pprint

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
from keras.regularizers import l2

class RankingModel(tfrs.Model):

    def __init__(self,user_features: tuple = (), anime_features: tuple = (), layer_sizes: list = [256, 64],
                    embedding_dims: int = 64, l2_regularization: float = 0.0, dropout: float = 0.0):
        super().__init__()

        self.user_features = user_features
        self.anime_features = anime_features
        self.user_model = UserModel(embedding_dims, self.user_features)
        self.anime_model = AnimeModel(embedding_dims, self.anime_features)


# Compute predictions.
        self.score_model = tf.keras.Sequential()

        assert (layer_sizes), "`config['layer_sizes']` cannot be empty list! Please ensure at least one layer is specified"

# Use the ReLU activation for all but the last layer
        if len(layer_sizes) > 1:
            for layer_size in layer_sizes:
                self.score_model.add(tf.keras.layers.Dense(layer_size, activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                kernel_regularizer=l2(l2_regularization)))
                self.score_model.add(tf.keras.layers.Dropout(dropout))
                #self.dense_layers.add(tf.keras.layers.BatchNormalization())

# Linear activation for last layer
        self.score_model.add(tf.keras.layers.Dense(1))#, kernel_regularizer=l2(l2_regularization)))

        self.task = tfrs.tasks.Ranking(
            loss=tfr.keras.losses.ListMLELoss(),
            metrics=[
            tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
            tf.keras.metrics.RootMeanSquaredError()
            ]
        )

    def call(self, features):
    # We first convert the id features into embeddings.
    # User embeddings are a [batch_size, embedding_dim] tensor.
        user_embeddings = self.user_model({
                'user_id': features['user_id'],
                **{k: features[k] for k in self.user_features if k in self.user_features}
            })

# Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
# tensor.
        anime_embeddings = self.anime_model({
                'title': features['title'],
                **{k: features[k] for k in self.anime_features if k in self.anime_features}
            })

# We want to concatenate user embeddings with movie emebeddings to pass
# them into the ranking model. To do so, we need to reshape the user
# embeddings to match the shape of movie embeddings.
        list_length = features["title"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

# Once reshaped, we concatenate and pass into the dense layers to generate
# predictions.
        if self.anime_features:
            anime_embeddings = tf.split(anime_embeddings, len(self.anime_features) + 1, axis=1)
            for anime_embedding in anime_embeddings:
                user_embedding_repeated = tf.concat(
                    [user_embedding_repeated, anime_embedding], 2)
                concatenated_embeddings = user_embedding_repeated
        else:
            concatenated_embeddings = tf.concat(
            [user_embedding_repeated, anime_embeddings], 2)

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, features, training=False):
        labels = features.pop("user_rating")

        scores = self(features)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(scores, axis=-1),
        )

r_model = RankingModel(user_features=("mean_score","completed_count","score_count"), anime_features=("popularity","episodes","format","year","studio","source","avg_score"), layer_sizes=[256,128,64], embedding_dims=64, dropout=0.01, l2_regularization=0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)#, beta_1=0.9, beta_2=0.999)
#tf.keras.optimizers.Adagrad(0.1)
r_model.compile(optimizer=optimizer, run_eagerly=True)

cached_train = train.shuffle(400_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
#cached_valid = validation.batch(512).cache()

early_stopping = tf.keras.callbacks.EarlyStopping(patience=1,
                                                  restore_best_weights=True, monitor="total_loss")

# Train model
history = r_model.fit(
    cached_train,
    #validation_data=cached_valid,
    #validation_freq=2,
    epochs=20,
    verbose='auto',
    callbacks=early_stopping
)

metrics = r_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(metrics["ndcg_metric"]))
