import pandas as pd
import requests
import json
import numpy as np

from tqdm import tqdm
from time import time, sleep
from datetime import datetime

QUERY_LIST = '''query ($uId:Int)  {
    MediaListCollection (userId:$uId, type:ANIME, status_in:[COMPLETED,DROPPED,PAUSED,CURRENT,REPEATING],forceSingleCompletedList:true) {
      user {
        id
        name
        createdAt
        updatedAt
        statistics {
          anime {
            count
            meanScore
            standardDeviation
            episodesWatched
            minutesWatched
            scores {
                count
            }
          }
        }
        favourites {
          anime1:anime (page:1) {
            nodes {
              id
            }
          }
          anime2:anime (page:2) {
            nodes {
              id
            }
          }
          anime3:anime (page:3) {
            nodes {
              id
            }
          }
          anime4:anime (page:4) {
            nodes {
              id
            }
          }
        }
      }
    lists {
        isCustomList
        name
        entries{
          score(format:POINT_100)
          progress
          status
          repeat
          completedAt {
              year
              month
              day
          }
          updatedAt
          createdAt
          media {
            id
          }
        }
      }
    }
}
'''

QUERY_LIST_SHORT = '''query ($uId:Int)  {
    MediaListCollection (userId:$uId, type:ANIME, status_in:[COMPLETED,DROPPED,PAUSED,CURRENT,REPEATING],forceSingleCompletedList:true) {
    lists {
        isCustomList
        name
        entries{
          score(format:POINT_100)
          progress
          status
          repeat
          completedAt {
              year
              month
              day
          }
          updatedAt
          createdAt
          media {
            id
        }
        }
      }
    }
}
'''

QUERY_USER_INFO = '''query ($uId:Int)  {
    User (id:$uId) {
        id
        name
        createdAt
        updatedAt
        statistics {
          anime {
            count
            meanScore
            standardDeviation
            episodesWatched
            minutesWatched
            scores {
                count
            }
          }
        }
    }
}
'''

FRAGMENT_INFO = '''fragment stuff on User{
  id
  name
  createdAt
  updatedAt
  statistics {
    anime {
      count
      minutesWatched
      episodesWatched
      meanScore
      standardDeviation
      scores {
        count
      }
    }
  }
}
'''

def get_season(season, month):
    """TODO: Docstring for get_season.

    :arg1: TODO
    :returns: TODO

    """
    if season is None:
        if month is None:
            return ''
        if 9 <= month < 12:
            return "FALL"
        elif 6 <= month < 9:
            return "SUMMER"
        elif 3 <= month < 6:
            return "SPRING"
        else:
            return "WINTER"
    else:
        return season

URL = 'https://graphql.anilist.co'

def try_query(query, variables, std_resp, delay):
    invalid = True
    count = 0
    while invalid:
        try:
            response = requests.post(URL, json={'query': query, 'variables':variables})
            query_dict = response.json()
            _ = query_dict["data"][std_resp]
            invalid = False
        except (TypeError, json.decoder.JSONDecodeError) as e:
            if count > 20:
                print("Count exceeded. Definite error, moving on")
                count = 0
                invalid = True
            else:
                print(f"Query limit reached. Retrying in {delay}s")
                print(query_dict)
                sleep(delay)
                count += 1

    if invalid:
        query_dict = {"data":{std_resp:None}}
    return query_dict

def paged_query(start_page, page_total, fragment):
    """TODO: Docstring for paged_query.

    :arg1: TODO
    :returns: TODO

    """
    query = "query{\n"
    for page in range(page_total):
        query += f"a{page+1}" + f":Page(page:{start_page + page})" + "{users{... stuff}}\n"
    query += "}\n"
    query += fragment
    return query

def user_info_scraping(start, stop, step, restart=False):
    """docstring for user_info_scraping"""
    df = pd.DataFrame(columns=["id", "name", "updated_at", "score_count", "watch_count",
                               "mean_score", "std_score", "minutes_watched",
                               "episodes_watched"])

    if restart:
        df = pd.read_csv("user_info.csv")

    for page in tqdm(range(start, stop, step)):
        query_info = paged_query(page, step, FRAGMENT_INFO)
        query_dict = try_query(query_info, "", "a1", 2)

        if query_dict["data"]["a1"] is None:
            print(f"Failed page: {page}")
            continue

        for users in query_dict["data"].values():
            for user in users["users"]:
                stat = user["statistics"]["anime"]
                df = pd.concat([df, pd.DataFrame([[
                    user["id"],
                    user["name"],
                    user["updatedAt"],
                    sum(d["count"] for d in stat["scores"]),
                    stat["count"],
                    stat["meanScore"],
                    stat["standardDeviation"],
                    stat["minutesWatched"],
                    stat["episodesWatched"],
                ]], columns=df.columns)])

        if page % max((stop // 100), 1) < step:
            df.to_csv("user_info.csv", columns=df.columns, index=False)

def user_info_light_scraping(start, stop, restart=False):
    """docstring for user_info_scraping"""
    df = pd.DataFrame(columns=["id", "name", "updated_at", "score_count", "watch_count",
                               "mean_score", "std_score", "minutes_watched",
                               "episodes_watched"])

    if restart:
        df = pd.read_csv("user_info_light.csv")

    user_list = pd.read_csv("user_list_curated.csv")
    user_ids = user_list[["id"]].iloc[start:stop].to_numpy().flatten()

    for num, user_id in enumerate(tqdm(user_ids)):
        user_id = int(user_id)
        query_dict = try_query(QUERY_USER_INFO, {"uId":user_id}, "User", 2)

        if query_dict["data"]["User"] is None:
            print(f"Failed collecting user_id : {user_id}")
            continue

        user = query_dict["data"]["User"]
        stat = user["statistics"]["anime"]
        df = pd.concat([df, pd.DataFrame([[
            user["id"],
            user["name"],
            user["updatedAt"],
            sum(d["count"] for d in stat["scores"]),
            stat["count"],
            stat["meanScore"],
            stat["standardDeviation"],
            stat["minutesWatched"],
            stat["episodesWatched"],
        ]], columns=df.columns)])

        if True or num % max((stop // 100), 1) < 1:
            df.to_csv("user_info_light.csv", columns=df.columns, index=False)

def user_list_scraping(start, stop, restart=False):
    """docstring for user_info_scraping"""
    user_df = pd.DataFrame(columns=[
        "user_id", "name", "user_creation", "user_updated_at", "score_count", "completed_count", "watch_count", "mean_score", "std_score", "minutes_watched",
        "episodes_watched"
    ])

    df = pd.DataFrame(columns=[
        "user_id", "media_id", "score", "status", "progress", "repeat", "is_fav",
        "created","updated", "completed", "completed_count"
    ])
    df["is_fav"] = df["is_fav"].astype("bool")

    # media_df = pd.DataFrame(columns=[
        # "media_id", "title_romaji", "title_english", "title_native", "id_Mal",
        # "episodes", "format", "status", "start_date", "year", "season", "duration",
        # "country", "source", "avg_score", "popularity", "is_licensed", "is_adult",
        # "favourites", "studio", "genres", "tags", "recommendations"
    # ])
    # media_df["is_licensed"] = media_df["is_licensed"].astype("bool")
    # media_df["is_adult"] = media_df["is_adult"].astype("bool")

    if restart:
        # df = pd.read_csv("anime_ratings.csv")
        # user_df = pd.read_csv("anime_user.csv")
        # media_df = pd.read_csv("anime_media.csv")
        pass

    user_list = pd.read_csv("user_list_curated.csv")
    user_ids = user_list[["id"]].iloc[start:stop].to_numpy().flatten()

    data = np.empty((len(user_ids)*750, 1), dtype=[
        ('user_id',np.int64), ('media_id',np.int64), ('score', np.int64), ('status', '<U25'), ('progress', np.int64),('repeat', np.int64),
        ('is_fav', np.bool_),('created', np.int64),('updated', np.int64),
        ('completed', '<U25')])#,('completed_count', np.int64)])

    # data_media = np.empty((10000, 1), dtype=[
        # ('media_id',np.int64), ('title_romaji','<U512'), ('title_english', '<U512'),
        # ('title_native', '<U512'), ('id_Mal',np.float64), ('episodes',np.float64),
        # ('format', '<U50'), ('status', '<U50'), ('start_date', '<U25'),
        # ('year', np.float64), ('season', '<U25'), ('duration', np.float64),
        # ('country', '<U25'), ('source', '<U25'), ('avg_score', np.float64),
        # ('popularity', np.int64), ('is_licensed', np.bool_), ('is_adult', np.bool_),
        # ('favourites', np.int64), ('studio', '<U25'), ('genres', '<U512'),
        # ('tags', '<U1024'), ('recommendations', '<U1024'),
    # ])

    data_user = np.empty((len(user_ids), 1), dtype=[
        ('user_id',np.int64), ('name','<U64'), ('user_creation', np.float64),
        ('user_updated_at', np.float64), ('score_count', np.int64),
        ('watch_count', np.int64), ('completed_count', np.int64),
        ('mean_score', np.float64), ('std_score', np.float64),
        ('minutes_watched', np.int64), ('episodes_watched', np.int64),
    ])
    entry = 0
    media_count = 0
    media_seen = set()

    for num, user_id in enumerate(tqdm(user_ids)):
        user_id = int(user_id)
        query_dict = try_query(QUERY_LIST, {'uId': user_id}, "MediaListCollection", 5)

        if query_dict["data"]["MediaListCollection"] is None:
            print(f"Failed collecting user_id : {user_id}")
            continue
        a = time()

        user_info = query_dict["data"]["MediaListCollection"]["user"]
        stat = user_info["statistics"]["anime"]
        favourites = set()
        for fav_entries in user_info["favourites"].values():
            favourites |= set(fav["id"] for fav in fav_entries["nodes"])

        for list_dict in query_dict["data"]["MediaListCollection"]["lists"]:
            if list_dict["name"].lower() == "completed":
                completed_count = len(list_dict["entries"])
            for per_entry_dict in list_dict["entries"]:
                media = per_entry_dict["media"]
                date = per_entry_dict["completedAt"]
                if date["year"] is None or date["month"] is None:
                    date = ''
                elif date["day"] is None:
                    try:
                        date = int(datetime(date["year"],date["month"],1).timestamp())
                    except ValueError:
                        date = ''
                else:
                    try:
                        date = int(datetime(date["year"],date["month"],date["day"]).timestamp())
                    except ValueError:
                        date = ''
                # df = pd.concat([df, pd.DataFrame([[
                data[entry] = (
                    user_id,
                    media["id"],
                    per_entry_dict["score"],
                    per_entry_dict["status"],
                    per_entry_dict["progress"],
                    per_entry_dict["repeat"],
                    media["id"] in favourites,
                    per_entry_dict["createdAt"],
                    per_entry_dict["updatedAt"],
                    date,
                    # completed_count,
                )
                # ]], columns=df.columns)])

                entry += 1

                # if not (media["id"] in media_seen or media_df[["media_id"]].isin([media["id"]]).any(axis=None)):
                    # date = media["startDate"]
                    # if date["year"] is None:
                        # date = ''
                    # elif date["month"] is None:
                        # date = int(datetime(date["year"],1,1).timestamp())
                    # elif date["day"] is None:
                        # date = int(datetime(date["year"],date["month"],1).timestamp())
                    # else:
                        # date = int(datetime(date["year"],date["month"],date["day"]).timestamp())
                    # season = get_season(media["season"], media["startDate"]["month"])
                    # studio = [studio["node"]["name"] for studio in media["studios"]["edges"] if studio["isMain"]]
                    # studio = studio[0] if studio else None
                    # # media_df = pd.concat([media_df, pd.DataFrame([[
                    # data_media[media_count] = (
                        # media["id"],
                        # media["title"]["romaji"],
                        # media["title"]["english"],
                        # media["title"]["native"],
                        # media["idMal"],
                        # media["episodes"],
                        # media["format"],
                        # media["status"],
                        # date,
                        # media["startDate"]["year"],
                        # season,
                        # media["duration"],
                        # media["countryOfOrigin"],
                        # media["source"],
                        # media["averageScore"],
                        # media["popularity"],
                        # media["isLicensed"],
                        # media["isAdult"],
                        # media["favourites"],
                        # studio,
                        # str(media["genres"]),
                        # str([(tag["name"], tag["rank"], tag["category"]) for tag in media["tags"]]),
                        # str([(rec["id"], rec["rating"]) for rec in media["recommendations"]["nodes"]]),
                    # )
                    # # ]], columns=media_df.columns)])

                    # media_count += 1
                    # media_seen.add(media["id"])

        # user_df = pd.concat([user_df, pd.DataFrame([[
        data_user[num] = (
            user_info["id"],
            user_info["name"],
            user_info["createdAt"],
            user_info["updatedAt"],
            completed_count,
            sum(d["count"] for d in stat["scores"]),
            stat["count"],
            stat["meanScore"],
            stat["standardDeviation"],
            stat["minutesWatched"],
            stat["episodesWatched"],
        )
        # ]], columns=user_df.columns)])

        if num % max(((stop-start) // 10), 1) < 1 or num == (stop-start - 1):
            df = pd.DataFrame(data[:entry,0])
            df.to_csv("anime_ratings.csv", index=False)
            user_df = pd.DataFrame(data_user[:(num+1),0])
            user_df.to_csv("anime_user.csv", index=False)
            # media_df = pd.concat([media_df, pd.DataFrame(data_media[:media_count,0])])
            # media_df.to_csv("anime_media.csv", index=False)
            # data_media = np.empty((10000, 1), dtype=[
                # ('media_id',np.int64), ('title_romaji','<U512'), ('title_english', '<U512'),
                # ('title_native', '<U512'), ('id_Mal',np.float64), ('episodes',np.float64),
                # ('format', '<U50'), ('status', '<U50'), ('start_date', '<U25'),
                # ('year', np.float64), ('season', '<U25'), ('duration', np.float64),
                # ('country', '<U25'), ('source', '<U25'), ('avg_score', np.float64),
                # ('popularity', np.int64), ('is_licensed', np.bool_), ('is_adult', np.bool_),
                # ('favourites', np.int64), ('studio', '<U25'), ('genres', '<U512'),
                # ('tags', '<U1024'), ('recommendations', '<U1024'),
            # ])
        duration = time() - a
        if  duration < 0.125:
            sleep(0.125 - duration)


if __name__ == "__main__":
    # user_info_scraping(10208, 37802, 25, restart=True)
    user_list_scraping(17000, 23212, restart=True)
    # user_list_scraping(697, 23212, restart=True)
    # user_info_light_scraping(3060, 3089)
