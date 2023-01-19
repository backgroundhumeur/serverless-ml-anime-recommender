# serverless-ml-anime-recommender
Project for the course ID2223.

This project consists of implementing a serverless Machine Learning pipeline
with the object of making daily predictions on a regularly updated data source.

The subject chosen is a recommender for [Anime](https://en.wikipedia.org/wiki/Anime), a term used to describe japanese animation,
using the openly available database with a [GraphQL API](https://anilist.co/graphiql?query=) from the anime aggregation website
<Anilist.co>.

This project was chosen because no other recommender of the sort, using
Tensorflow's recommender models, exists right now, and there are no known
available datasets of the Anilist website up on the internet as of now.

## The data & features

The historical data used for the training of the model consists of 7.2 million
user ratings taken from 23212 users of the website that have been curated from
the entire userbase, as representative and properly discriminant members in the
hope of a better and cleaner data to learn from.

The data in itself is primarily composed of a user id, an anime title and a user
rating as well as some other secondary features.

It has been collected using the API with the scripts in `data_query.py`.

The resulting raw dataset is a 1.6GB .csv file.

The secondary features retained were the following :

- For the user : their mean score, their number of scores and their completed
    titles count
- For the anime : their popularity, their average score on the website, their
    year of creation, the studio that created them, their number of episodes, their
    format(movie or TV series) and their source(book if an adaptation or
    original)

## The model

The model consists of two separate models that used consecutively to enact the
recommendation process. 

First their is a two tower Retrieval model, that takes
in as a query tower the user and its secondary features, and takes as
a candidate tower the anime's title and its other features. 

The retrieval part permits us to get a pool of suitable candidate to recommend
from.

Then comes the ranking part with a Ranking model, that handles the tast of
predicting from a user list and a given title and their other features, the
score given to that title.

The ranking model will rank all the titles that we retrieved earlier and we can
display the top results for out recommendation.

Both model are multi-layered deep neural networks with a feed forward
architecture but since there wasn't an extensive hyperparameter search, their
individual parameters won't be mentioned in great detail here.

The current overall accuracy of both models is first for the Retrival of 0.14
using factorized topK metric for the top 100, and for the listwise Ranking it is
of 0.73 using the NDCG metric that takes into account the order of every entry
on the list in comparison to the others. The training was only done on 2 million
of the ratings for the first and 400k for the second.

The current performances can definitely be improved using sharper and more
extensive hyperparameter tuning, as well as increasing the input data.

## The pipeline

The current pipeline consists of two feature and two training pipelines that
handle respectively the feature processing and the training of the model on
those features. The features are first computed then stored on GDrive to then be
downloaded by the training pipelines.

The models are to be stored on a huggingface repository and able to be loaded
anywhere.

This segmented pipeline allows us to very easily add new features to perform
daily batch inference on activity data(see below) and also to keep our feature
up to date with the constant flow of new releases and re-discovery of classical
works.

In the future one improvement would be to make use of Hopsworks' feature store
that easily allows us to select the features that we want and see real-time
performances of the model with the constant influx of new features.

## The daily prediction service

For now, the daily prediction service consists of estimating ratings collected
daily using the activity feature of <Anilist.co> that displays what users have
completed and scored on a day-to-day basis. We are limiting our activity scraping to the 23k users that we selected earlier as they are all currently active users(i.e. updated their list in the last two weeks).

We use batch inference to perform the prediction and then compare it to what the
score really was.

Those new results are also entered in the feature database to be possibly used in the
next training of the model.

In the future, the main idea is to set up a recommendation service that allows
user to enter their list and get recommended titles from the model, with
additional options like first selecting which titles to recommend from in the
first place, and then giving the choice of filtering the retrieval results.
This would allow users to more tailor the results to their specific needs, like
wanting to get recommend a movie that is not too popular, and so on.
