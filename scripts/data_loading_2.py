import random
import requests
import os
import pandas as pd
from timeit import default_timer as timer


def float_seconds_to_time_str(seconds, decimal_places_to_round_to):
    if seconds < 60.0:
        time = f"{round(seconds, decimal_places_to_round_to)} seconds"
    elif seconds / 60.0 < 60.0:
        time = f"{round(seconds / 60.0, decimal_places_to_round_to)} minutes"
    else:
        time = f"{round((seconds / 60.0) / 60.0, decimal_places_to_round_to)} hours"
    return time


def get_reviews(app_id, params):
    url = 'https://store.steampowered.com/appreviews/'
    response = requests.get(url=url+str(app_id), params=params, headers={'User-Agent': 'Mozilla/5.0'})
    return response.json()


def get_n_reviews(appid, n=100, cursor='*'):
    reviews = []
    params = {
            'json': 1,
            'filter': 'updated',
            'language': 'english'
            }
    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100
        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        if len(response['reviews']) < 100:
            no_more_reviews = True
            break
    return reviews, cursor


def get_last_reviews_dataset_id(path):
    subdirectory_names = [int(name) for name in os.listdir(path) if
                          os.path.isdir(path + "/" + name) and name.isdigit()]
    last_reviews_dataset_id = -1
    if len(subdirectory_names) > 0:
        last_reviews_dataset_id = max(subdirectory_names)
    return last_reviews_dataset_id

# 1245620 # Elden Ring
# 1949770 # Random small game


review_datasets_storage_path = "../data/reviews"
reviews_dataset_id = get_last_reviews_dataset_id(review_datasets_storage_path) + 1
path_to_save_reviews = review_datasets_storage_path + '/' + str(reviews_dataset_id)
os.makedirs(path_to_save_reviews, exist_ok=True)

# app_ids = [1245620, 1949770]
app_ids = [2337640, 1949770, 2357570, 570, 201510, 792990, 2016940, 247950,
           374320, 367520, 435150, 1888160, 814380, 730, 588650,
           292030, 570940, 236430]

voted_up_frac = 0.5

reviews_to_get = 1000000
checkpoint_n = 1000

start = timer()
cursors = ['*' for i in app_ids]
n = reviews_to_get

pd.DataFrame({'reviews_to_get': [reviews_to_get],
              'checkpoint_n': [checkpoint_n],
              'voted_up_frac': [voted_up_frac],
              'start_time': [start]}).to_csv(f"{path_to_save_reviews}/description.csv", index=False)

pd.DataFrame({'app_ids': app_ids, 'cursors': cursors}).to_csv(f"{path_to_save_reviews}/apps_description.csv", index=False)

while n > 0:
    index = random.randint(0, len(app_ids) - 1)
    app_id = app_ids[index]

    reviews, cursors[index] = get_n_reviews(app_id, min(checkpoint_n, n), cursor=cursors[index])
    reviews_df = pd.DataFrame()
    if len(reviews) > 0:
        reviews_df = pd.DataFrame(reviews)[['review', 'voted_up']]

        ####### Data aligment #######
        vu_frac = reviews_df['voted_up'].value_counts(True)[True]
        vd_frac = reviews_df['voted_up'].value_counts(True)[False]
        if vu_frac >= voted_up_frac:
            frac_to_drop = 1 - (voted_up_frac * vd_frac) / (vu_frac - voted_up_frac * vu_frac)
        else:
            frac_to_drop = 1 - (vu_frac - vu_frac * voted_up_frac) / (vd_frac * voted_up_frac)

        voted_up_reviews = reviews_df['voted_up'] == (vu_frac >= voted_up_frac)
        voted_up_to_drop = reviews_df[voted_up_reviews].sample(frac=frac_to_drop)
        reviews_df = reviews_df.drop(voted_up_to_drop.index)
        ####### ------------- #######
        end = timer()

        num_loaded = (reviews_to_get - (n - reviews_df.shape[0]))
        remaining_time = end - start
        eta = (remaining_time / num_loaded) * n
        print(f"{num_loaded}/{reviews_to_get}   "
              f"elapsed time: {float_seconds_to_time_str(remaining_time, 2)}     "
              f"ETA: {float_seconds_to_time_str(eta, 2)}    "
              f"apps left: {len(app_ids)}")

        reviews_df.to_csv(f"{path_to_save_reviews}/reviews.csv", index=False, mode='a', header=(n == reviews_to_get))

        log = pd.DataFrame({'app_id': [app_id],
                            'cursor': [cursors[index]],
                            'timestamp': [end]})
        log.to_csv(f"{path_to_save_reviews}/log.csv", index=False, mode='a', header=(n == reviews_to_get))
        del log

    if len(reviews) < min(checkpoint_n, n) or len(reviews) <= 0:
        if len(reviews) <= 0:
            print('len(reviews) <= 0')
        app_ids.pop(index)
        cursors.pop(index)
        print(f"No reviews left in the app with id {app_id}.")
        if len(app_ids) <= 0:
            print(f"There are no reviews left. Downloading has been completed.")
            break

    n -= reviews_df.shape[0]
    del reviews_df
