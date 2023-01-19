import pandas as pd

def main():
    """TODO: Docstring for main.
    """
    ratings_df = pd.read_csv('dataset/anime_ratings.csv')
    users = pd.read_csv('dataset/anime_user.csv')
    media = pd.read_csv('dataset/anime_media.csv')
    media = media.rename(columns={"status":"media_status"})

    users = users[["user_id","score_count","mean_score","completed_count"]]
    media = media[["media_id","title_romaji","episodes","format","media_status","duration","source","season","year","country","avg_score","popularity","is_adult","studio","genres"]]
    ratings_df = ratings_df[["user_id","media_id","score","status","is_fav","repeat","updated","completed"]]

    anime_ratings = ratings_df.merge(users, how="outer")
    anime_ratings = anime_ratings.merge(media, how="outer")
    anime_ratings = anime_ratings[anime_ratings["score"] > 0]

    base_avg_score = anime_ratings.groupby("media_id", as_index=False).mean().reset_index()["avg_score"].mean()
    anime_ratings = anime_ratings.fillna(value={"year":1950,"episodes":1,"mean_score":75.0, "score_count":251, "completed_count":242,"popularity":100})
    anime_ratings = anime_ratings.astype({"year":"int64","episodes":"int64"})
    anime_ratings["mean_score"] = anime_ratings["mean_score"] * 100
    anime_ratings = anime_ratings.astype({"mean_score":"int64"})
    anime_ratings = anime_ratings.fillna(value={"studio":"Unknown","avg_score":base_avg_score,"source":"Unknown", "format":"Unknown", "is_adult":False})
    anime_ratings["avg_score"] = anime_ratings["avg_score"] * 100
    anime_ratings = anime_ratings.astype({"avg_score":"int64"})
    anime_ratings = anime_ratings.dropna(subset=["title_romaji"])
    anime_ratings = anime_ratings.astype({"user_id":"int64"})
    anime_ratings = anime_ratings.astype({"media_id":"int64"})
    anime_ratings = anime_ratings.astype({"score":"float32"})
    anime_ratings = anime_ratings.astype({"repeat":"int64"})
    anime_ratings = anime_ratings.astype({"score_count":"int64"})
    anime_ratings = anime_ratings.astype({"completed_count":"int64"})
    anime_ratings = anime_ratings.astype({"popularity":"int64"})
    anime_df = anime_ratings[["title_romaji","episodes","format","year","source","popularity","avg_score","is_adult","studio"]].drop_duplicates("title_romaji").rename(columns={"title_romaji":"title"})

    anime_df.to_csv("dataset/processed_anime.csv")
    anime_ratings.to_csv("dataset/processed_anime_ratings.csv")

if __name__ == "__main__":
    main()
