import requests
import pandas as pd
import keyboard
import random
import numpy as np
from timeit import default_timer as timer
import os


def float_seconds_to_time_str(seconds, decimal_places_to_round_to):
    if seconds < 60.0:
        time = f"{round(seconds, decimal_places_to_round_to)} seconds"
    elif seconds / 60.0 < 60.0:
        time = f"{round(seconds / 60.0, decimal_places_to_round_to)} minutes"
    else:
        time = f"{round((seconds / 60.0) / 60.0, decimal_places_to_round_to)} hours"
    return time


def to_percent_encoding(input_str: str):
    input_str = input_str.replace('␣', '%20')
    input_str = input_str.replace('!', '%21')
    input_str = input_str.replace('"', '%22')
    input_str = input_str.replace('#', '%23')
    input_str = input_str.replace('$', '%24')
    input_str = input_str.replace('%', '%25')
    input_str = input_str.replace('&', '%26')
    input_str = input_str.replace("'", '%27')
    input_str = input_str.replace("(", '%28')
    input_str = input_str.replace(")", '%29')
    input_str = input_str.replace("*", '%2A')
    input_str = input_str.replace("+", '%2B')
    input_str = input_str.replace(",", '%2C')
    input_str = input_str.replace("/", '%2F')
    input_str = input_str.replace(":", '%3A')
    input_str = input_str.replace(";", '%3B')
    input_str = input_str.replace("=", '%3D')
    input_str = input_str.replace("?", '%3F')
    input_str = input_str.replace("@", '%40')
    input_str = input_str.replace("[", '%5B')
    input_str = input_str.replace("]", '%5D')
    return input_str


class SteamWorks:
    def __init__(self):
        self.reviews_loading_force_stop = False
        self.reviews_downloading_to_csv_force_stop = False
        pass

    # @staticmethod
    # def get_url(app_id, num_per_page=100, language='all', cursor='*', **kwargs):
    #     return f"https://store.steampowered.com/appreviews/{app_id}?" \
    #            f"json=1&" \
    #            f"num_per_page={num_per_page}&" \
    #            f"language={language}&" \
    #            f"cursor={cursor}"

    @staticmethod
    def get(app_id, command='appreviews', num_per_page=100, language='english', filter_='updated', cursor='*', **kwargs):
        url = f"https://store.steampowered.com/{command}/{app_id}?" \
               f"json=1&" \
               f"filter={filter_}&" \
               f"num_per_page={num_per_page}&" \
               f"language={language}&" \
               f"cursor={cursor}"
        return requests.get(url)

    def stop_loading_reviews(self, e):
        self.reviews_loading_force_stop = True

    def stop_downloading_reviews_to_csv(self, e):
        self.reviews_downloading_to_csv_force_stop = True

    def load_reviews(self, app_id,
                     columns=['recommendationid', 'review', 'voted_up'],
                     cursor='*',
                     max_reviews_in_memory=10000,
                     downloaded_reviews_count=0,
                     verbose=1,
                     **kwargs):
        cursors = [cursor]
        response = SteamWorks.get(app_id)
        total_reviews = -1
        if 'total_reviews' in response.json()['query_summary']:
            total_reviews = response.json()['query_summary']['total_reviews']
        current_total_reviews = 0
        rows = []
        num_per_page = 100

        keyboard.on_press_key("c", self.stop_loading_reviews)
        found_duplicate_cursor = False
        while True:
            if self.reviews_loading_force_stop or self.reviews_downloading_to_csv_force_stop:
                if verbose != 0:
                    print('Loading was forcefully stopped')
                self.reviews_loading_force_stop = False
                break

            if current_total_reviews >= max_reviews_in_memory:
                if verbose != 0:
                    print(f'Maximum number of reviews ({max_reviews_in_memory}) in memory has been exceeded.')
                break
            
            if current_total_reviews + num_per_page > max_reviews_in_memory:
                break
            
            response = SteamWorks.get(app_id, num_per_page=num_per_page, cursor=to_percent_encoding(cursor))
            response_json = response.json()
            success = response_json['success']
            if success == 1:
                cursor = response_json['cursor']

                if cursor in cursors:
                    print(f'(app {app_id}) Found a duplicate cursor: {cursor}')
                    found_duplicate_cursor = True
                    break
                cursors.append(cursor)

                num_reviews = response_json['query_summary']['num_reviews']
                current_total_reviews += num_reviews
                downloaded_reviews_count += num_reviews
                if verbose != 0:
                    print(f'    downloaded: {downloaded_reviews_count}/{total_reviews}, '
                        f'reviews_in_memory: {current_total_reviews}, '
                        f'cursor: {cursor}')
                for j in range(num_reviews):
                    
                    new_row = {}
                    for k in range(len(columns)):
                        new_row[columns[k]] = response_json['reviews'][j][columns[k]]
                    rows.append(new_row)

                if current_total_reviews >= total_reviews:
                    
                    break
            else:
                error = response_json["error"]
                print(f'Request failed with the following error: "{error}"')
                break

        data_frame = pd.DataFrame(rows, columns=['app_id', 'recommendationid', 'review', 'voted_up'])
        return {'data_frame': data_frame,
                'last_cursor': cursors[-1],
                'downloaded_reviews_count': current_total_reviews,
                'found_duplicate_cursor': found_duplicate_cursor
                }

    def get_total_reviews_count(self, app_id):
        return self.get(app_id).json()['query_summary']['total_reviews']

    def get_up_votes_proportions(self, app_ids: list, max_reviews_in_memory=1000, verbose=1):
        voted_up_proportions = []

        total_elapsed_time = 0
        for i, app_id in enumerate(app_ids):
            start = timer()
            rd = self.load_reviews(app_id,
                                   columns=['voted_up'],
                                   verbose=0,
                                   max_reviews_in_memory=max_reviews_in_memory,
                                   downloaded_reviews_count=0)
            reviews_dataframe = rd['data_frame']
            voted_up_proportion = reviews_dataframe['voted_up'].value_counts(True)[True]
            voted_up_proportions.append(voted_up_proportion)
            end = timer()
            elapsed_time = end - start
            total_elapsed_time += elapsed_time
            eta = (total_elapsed_time / (i + 1)) * (len(app_ids) - (i + 1))
            print(f"{i + 1}/{len(app_ids)} ETA: {float_seconds_to_time_str(eta, 2)}")
        return voted_up_proportions

    @staticmethod
    def get_up_vote_proportions_to_keep(up_votes_proportions: list, desired_portion):
        up_vote_proportions_to_keep = []

        for up_vote_portion in up_votes_proportions:
            if up_vote_portion > 0.5:
                down_vote_portion = 1 - up_vote_portion
                up_vote_portion_to_keep = (desired_portion * down_vote_portion) / (up_vote_portion * (1 - desired_portion))
                up_vote_proportions_to_keep.append(up_vote_portion_to_keep)
            else:
                up_vote_proportions_to_keep.append(1)
        return up_vote_proportions_to_keep

    @staticmethod
    def get_down_vote_proportions_to_keep(up_vote_proportions: list, desired_portion):
        down_vote_proportions = [(1 - proportion) for proportion in up_vote_proportions]
        return SteamWorks.get_up_vote_proportions_to_keep(down_vote_proportions, 1 - desired_portion)

    @staticmethod
    def drop_frac_of_up_voted_rows(reviews_dataframe, up_votes_frac):
        up_vote_entries_percentage_to_drop = up_votes_frac
        is_up_voted_rows = reviews_dataframe['voted_up'] == True
        rows_to_drop = reviews_dataframe[is_up_voted_rows].sample(frac=up_vote_entries_percentage_to_drop)
        reviews_dataframe = reviews_dataframe.drop(rows_to_drop.index)
        return reviews_dataframe

    @staticmethod
    def drop_frac_of_down_voted_rows(reviews_dataframe, down_votes_frac):
        down_vote_entries_percentage_to_drop = down_votes_frac
        is_down_voted_rows = reviews_dataframe['voted_up'] == False
        rows_to_drop = reviews_dataframe[is_down_voted_rows].sample(frac=down_vote_entries_percentage_to_drop)
        reviews_dataframe = reviews_dataframe.drop(rows_to_drop.index)
        return reviews_dataframe

    def download_reviews_to_csv_several_ids(self, path,
                                            app_id: list,
                                            columns=['recommendationid', 'review', 'voted_up'],
                                            cursor="*",
                                            max_reviews_in_memory=3000,
                                            desired_up_votes_proportion=0.5,
                                            **kwargs):
        print(f"Downloading reviews from {app_id}")
        print(f'Estimating fractions of rows to drop: ')
        up_votes_proportions_per_app = self.get_up_votes_proportions(app_id, max_reviews_in_memory=1000)
        up_vote_proportions_to_keep = self.get_up_vote_proportions_to_keep(up_votes_proportions_per_app, desired_up_votes_proportion)
        down_vote_proportions_to_keep = self.get_down_vote_proportions_to_keep(up_votes_proportions_per_app, desired_up_votes_proportion)

        keyboard.on_press_key("q", self.stop_downloading_reviews_to_csv)
        last_cursors = np.full(shape=len(app_id), fill_value="*").tolist()
        downloaded_reviews_count = 0
        downloaded_reviews_count_per_app = np.full(shape=len(app_id), fill_value=0)
        downloading_completed = np.full(shape=len(app_id), fill_value=False).tolist()
        
        total_reviews_count = 0
        total_reviews_count_per_app = []
        for app_id_index in range(len(app_id)):
            if up_votes_proportions_per_app[app_id_index] > 0.5:
                up_voted_frac_to_discard = up_votes_proportions_per_app[app_id_index] * \
                                           (1 - up_vote_proportions_to_keep[app_id_index])
            else:
                up_voted_frac_to_discard = (1 - up_votes_proportions_per_app[app_id_index]) * \
                                           (1 - down_vote_proportions_to_keep[app_id_index])

            reviews_count = self.get_total_reviews_count(app_id[app_id_index]) * (1 - up_voted_frac_to_discard)
            total_reviews_count_per_app.append(int(reviews_count))
            total_reviews_count += total_reviews_count_per_app[-1]
        
        max_reviews_in_memory_per_app = int(max_reviews_in_memory / len(app_id))
        
        elapsed_time = 0
        reviews_data = np.full(shape=len(app_id), fill_value=0).tolist()
        i = 0
        while True:
            if self.reviews_downloading_to_csv_force_stop:
                self.reviews_downloading_to_csv_force_stop = False
                print('Downloading reviews to csv was forcefully stopped.')
                break

            for x in range(len(app_id)):
                if not self.reviews_downloading_to_csv_force_stop:
                    if not downloading_completed[x]:
                        start = timer()
                        rd = self.load_reviews(app_id[x],
                                                columns=columns,
                                                cursor=last_cursors[x],
                                                verbose=0,
                                                max_reviews_in_memory=max_reviews_in_memory_per_app,
                                                downloaded_reviews_count=downloaded_reviews_count_per_app[x])
                        if up_vote_proportions_to_keep[x] < 1:
                            rd['data_frame'] = SteamWorks.drop_frac_of_up_voted_rows(rd['data_frame'],
                                                                                     1 - up_vote_proportions_to_keep[x])
                        else:
                            rd['data_frame'] = SteamWorks.drop_frac_of_down_voted_rows(rd['data_frame'],
                                                                                        1 - down_vote_proportions_to_keep[x])
                        reviews_data[x] = rd

                        # downloaded_reviews_count_per_app[x] += reviews_data[x]['downloaded_reviews_count']
                        # downloaded_reviews_count += reviews_data[x]['downloaded_reviews_count']
                        downloaded_reviews_count_per_app[x] += reviews_data[x]['data_frame'].size
                        downloaded_reviews_count += reviews_data[x]['data_frame'].size
                        last_cursors[x] = reviews_data[x]['last_cursor']
                        
                        reviews_data[x]['data_frame'].to_csv(f'{path}/reviews.csv',
                                        index=False, 
                                        mode=('w' if not i else 'a'), 
                                        header=(not i))
                        del reviews_data[x]['data_frame']
                        
                        end = timer()
                        elapsed_time += (end - start)
                        
                        time_left = (elapsed_time / downloaded_reviews_count) * total_reviews_count - elapsed_time
                        

                        
                        print(f"{downloaded_reviews_count}/~{total_reviews_count}    "
                              f"app reviews downloaded: {downloaded_reviews_count_per_app[x]}/~{total_reviews_count_per_app[x]}   " 
                              f"last cursor: {last_cursors[x]}    app: {app_id[x]}    "
                              f"elapsed time: {float_seconds_to_time_str(elapsed_time, 2)}     "
                              f"ETA: {float_seconds_to_time_str(time_left, 2)}")
                        
                        
                        if (downloaded_reviews_count_per_app[x] >= total_reviews_count_per_app[x] or 
                            reviews_data[x]['found_duplicate_cursor'] or
                            not (False in downloading_completed)):
                            
                            downloading_completed[x] = True
                            print("\n" + "-" * 30)
                            print(f"Downloading reviews from {app_id[x]} has been completed:"
                                f"{downloaded_reviews_count_per_app[x]}/~{total_reviews_count_per_app[x]}.")
                            print("-" * 30 + "\n")

            if downloaded_reviews_count >= total_reviews_count or not (False in downloading_completed):
                break
            i += 1

        description_dataframe = pd.DataFrame({'columns': [columns],
                                              'app_ids': str(app_id),
                                              'desired_up_votes_proportion': [desired_up_votes_proportion],
                                              'downloaded_reviews_count': [downloaded_reviews_count],
                                              'downloading_time': [elapsed_time],
                                              'last_cursors': str(last_cursors)})
        description_dataframe.to_csv(f'{path}/description.csv', index=False)
        print(f"Downloading has been completed ({downloaded_reviews_count}/{total_reviews_count}). Yay!! ^-^")


# TODO:
# Make the code more readable*
# Add the ability to select the ration of upvotes and downvotes
# Save reviews to a folder with additional description of the information in a csv file


def get_last_reviews_dataset_id(path):
    subdirectory_names = [int(name) for name in os.listdir(path) if
                          os.path.isdir(path + "/" + name) and name.isdigit()]
    last_reviews_dataset_id = -1
    if len(subdirectory_names) > 0:
        last_reviews_dataset_id = max(subdirectory_names)
    return last_reviews_dataset_id


if __name__ == '__main__':
    # app_id = 1245620    # Elden Ring
    app_ids = [1245620, 570, 292030, 570940, 236430, 374320, 367520, 435150, 1888160, 814380, 730, 588650]

    review_datasets_storage_path = "../data/reviews"
    reviews_dataset_id = get_last_reviews_dataset_id(review_datasets_storage_path) + 1
    path_to_save_reviews = review_datasets_storage_path + '/' + str(reviews_dataset_id)
    os.makedirs(path_to_save_reviews, exist_ok=True)

    steam_works = SteamWorks()
    steam_works.download_reviews_to_csv_several_ids(path_to_save_reviews, app_ids)
