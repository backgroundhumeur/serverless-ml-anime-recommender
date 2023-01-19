from typing import Dict, Text, Union
from keras.regularizers import l2

import tensorflow as tf
import tensorflow_recommenders as tfrs

class ModelStack(tf.keras.Model):

    def __init__(self, embedding_model: Union[UserModel, AnimeModel], layer_sizes: list = [64, 32],
                 l2_regularization: float = 0.0, dropout: float = 0.0):
        """Model stack for user or product model

        Args:
            embedding_model, instance of a class, either User or Product model class
            layer_sizes: A list of integers where the i-th entry represents the number of units the i-th layer contains
            l2_regularization: float, l2 regularization term to apply to kernel_regularizer dense layers, values should
                range between 0-0.1.
        """
        super().__init__()

        # Initialize user model for embeddings
        self.embedding_model = embedding_model

        # Initialize model to add layers
        self.dense_layers = tf.keras.Sequential()

        assert (layer_sizes), "`config['layer_sizes']` cannot be empty list! Please ensure at least one layer is specified"

        # Use the ReLU activation for all but the last layer
        if len(layer_sizes) > 1:
            for layer_size in layer_sizes[:-1]:
                self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                                            kernel_regularizer=l2(l2_regularization)))
                self.dense_layers.add(tf.keras.layers.Dropout(dropout))
                #self.dense_layers.add(tf.keras.layers.BatchNormalization())

        # Linear activation for last layer
        self.dense_layers.add(tf.keras.layers.Dense(layer_sizes[-1], kernel_regularizer=l2(l2_regularization)))

    def call(self, inputs) -> tf.Tensor:
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class RetrievalModel(tfrs.models.Model):

    def __init__(self, anime, user_features: tuple = (), anime_features: tuple = (), layer_sizes: list = [64, 32],
                    embedding_dims: int = 64, l2_regularization: float = 0.0, dropout: float = 0.0):
        super().__init__()

        self.user_features = user_features
        self.anime_features = anime_features
        self.user_model = UserModel(embedding_dims, self.user_features)
        self.anime_model = AnimeModel(embedding_dims, self.anime_features)
        self.anime = anime
        self.k = 100

        self.query_model = ModelStack(self.user_model, layer_sizes, l2_regularization, dropout)
        self.candidate_model = ModelStack(self.anime_model, layer_sizes, l2_regularization, dropout)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=(self.anime.batch(128).map(self.candidate_model)),
            ),
        )

    def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        query_embeddings = self.query_model({
                'user_id': features['user_id'],
                **{k: features[k] for k in self.user_features if k in self.user_features}
            })
        anime_embeddings = self.candidate_model({
                'title': features['title'],
                **{k: features[k] for k in self.anime_features if k in self.anime_features}
            })

        return self.task(
            query_embeddings, anime_embeddings, compute_metrics=not training)

    def call(self, inputs):
        index = tfrs.layers.factorized_top_k.BruteForce(self.query_model, k=self.k)
        index.index_from_dataset(
        tf.data.Dataset.zip((self.anime.map(lambda x: x['title']).batch(128), self.anime.batch(128).map(self.candidate_model)))
        )

        _, titles = index(inputs)
        return titles

model = RetrievalModel(anime, user_features=("mean_score","score_count","completed_count"), anime_features=("popularity","episodes","format","year","studio","source","avg_score"), layer_sizes=[128,64], embedding_dims=64, dropout=0.1, l2_regularization=0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)#, beta_1=0.9, beta_2=0.999)
#tf.keras.optimizers.Adagrad(0.1)
model.compile(optimizer=optimizer, run_eagerly=True)
cached_train = train.shuffle(1_750_000).batch(512).cache()
cached_test = test.batch(512).cache()
cached_valid = validation.batch(512).cache()

early_stopping = tf.keras.callbacks.EarlyStopping(patience=1,
                                                  restore_best_weights=True, monitor="total_loss")

# Train model
history = model.fit(
    cached_train,
    validation_data=cached_valid,
    validation_freq=2,
    epochs=5,
    verbose='auto',
    callbacks=early_stopping
)

metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
