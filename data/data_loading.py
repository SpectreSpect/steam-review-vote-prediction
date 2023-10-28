import requests
import pandas as pd
import keyboard
import random
import numpy as np
from timeit import default_timer as timer

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

    def download_reviews(self, app_id,
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


    def download_reviews_to_csv(self, path, 
                                app_id,
                                columns=['recommendationid', 'review', 'voted_up'], 
                                cursor="*", 
                                max_reviews_in_memory=10000, 
                                **kwargs):
        file_id = random.randint(0, 10 ** 10)

        keyboard.on_press_key("q", self.stop_downloading_reviews_to_csv)
        last_cursor = cursor
        total_downloaded_reviews_count = 0
        total_reviews = self.get_total_reviews_count(app_id)
        total_elapsed_time = 0
        i = 0
        while True:
            if self.reviews_downloading_to_csv_force_stop:
                self.reviews_downloading_to_csv_force_stop = False
                print('Downloading reviews to csv was forcefully stopped.')
                break

            start = timer()
            reviews_data = self.download_reviews(app_id,
                                                 columns=columns,
                                                 cursor=last_cursor,
                                                 max_reviews_in_memory=max_reviews_in_memory,
                                                 downloaded_reviews_count=total_downloaded_reviews_count)
            end = timer()
            elapsed_time = end - start
            
            total_elapsed_time += elapsed_time
            data_frame = reviews_data['data_frame']
            total_downloaded_reviews_count += reviews_data['downloaded_reviews_count']
            last_cursor = reviews_data['last_cursor']
            
            print(f'Saving data... ')
            print(f"Reviews downloaded: {total_downloaded_reviews_count}/{total_reviews}, "
                  f"last_cursor: {last_cursor}, "
                  f"elapsed_time: {elapsed_time}, "
                  f"total_elapsed_time: {total_elapsed_time}")
            
            data_frame.to_csv(f'{path}/app_id{app_id}file_id{file_id}.csv', 
                                index=False, 
                                mode=('w' if not i else 'a'), 
                                header=(not i))
            del data_frame

            if total_downloaded_reviews_count >= total_reviews:
                print(f"Downloading has been completed ({total_downloaded_reviews_count}/{total_reviews}). Yay!! ^-^")
                break
            i += 1
        print(f"Reviews downloaded: {total_downloaded_reviews_count}/{total_reviews}, "
              f"Last cursor: {last_cursor}")
    
    
    def get_total_reviews_count(self, app_id):
        return self.get(app_id).json()['query_summary']['total_reviews']
    
    
    def download_reviews_to_csv_several_ids(self, path, 
                                app_id: list,
                                columns=['recommendationid', 'review', 'voted_up'], 
                                cursor="*", 
                                max_reviews_in_memory=3000, 
                                **kwargs):
        apps_count = len(app_id)
        print("Downloading reviews from ", end="")
        for i in range(apps_count):
            if i < apps_count - 1:
                print(f"{app_id[i]}, ", end="")
            else:
                print(f"{app_id[i]}.")
        
        file_id = random.randint(0, 10 ** 10)

        keyboard.on_press_key("q", self.stop_downloading_reviews_to_csv)
        last_cursors = np.full(shape=len(app_id), fill_value="*").tolist()
        downloaded_reviews_count = 0
        downloaded_reviews_count_per_app = np.full(shape=len(app_id), fill_value=0)
        downloading_completed = np.full(shape=len(app_id), fill_value=False).tolist()
        
        total_reviews_count = 0
        total_reviews_count_per_app = []
        for id in app_id:
            total_reviews_count_per_app.append(self.get_total_reviews_count(id))
            total_reviews_count += total_reviews_count_per_app[-1]
        
        max_reviews_in_memory_per_app = max_reviews_in_memory / len(app_id)
        
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
                        rd = self.download_reviews(app_id[x],
                                                columns=columns,
                                                cursor=last_cursors[x],
                                                verbose=0,
                                                max_reviews_in_memory=max_reviews_in_memory_per_app,
                                                downloaded_reviews_count=downloaded_reviews_count_per_app[x])
                        reviews_data[x] = rd
                        
                        downloaded_reviews_count_per_app[x] += reviews_data[x]['downloaded_reviews_count']
                        downloaded_reviews_count += reviews_data[x]['downloaded_reviews_count']
                        last_cursors[x] = reviews_data[x]['last_cursor']
                        
                        reviews_data[x]['data_frame'].to_csv(f'{path}/app_id{app_id}file_id{file_id}.csv', 
                                        index=False, 
                                        mode=('w' if not i else 'a'), 
                                        header=(not i))
                        del reviews_data[x]['data_frame']
                        
                        end = timer()
                        elapsed_time += (end - start) / 60.0
                        
                        time_left = (elapsed_time / downloaded_reviews_count) * total_reviews_count - elapsed_time
                        
                        print(f"{downloaded_reviews_count}/{total_reviews_count}    "
                            f"app reviews downloaded: {downloaded_reviews_count_per_app[x]}/{total_reviews_count_per_app[x]}   " 
                            f"last cursor: {last_cursors[x]}    app: {app_id[x]}    "
                            f"elapsed time: {round(elapsed_time, 2)} min    "
                            f"~{round(time_left, 2)} minutes left")
                        
                        
                        if (downloaded_reviews_count_per_app[x] >= total_reviews_count_per_app[x] or 
                            reviews_data[x]['found_duplicate_cursor'] or
                            not (False in downloading_completed)):
                            
                            downloading_completed[x] = True
                            print("\n" + "-" * 30)
                            print(f"Downloading reviews from {app_id[x]} has been completed:"
                                f"{downloaded_reviews_count_per_app[x]}/{total_reviews_count_per_app[x]}.")
                            print("-" * 30 + "\n")

            if downloaded_reviews_count >= total_reviews_count or not (False in downloading_completed):
                break
            i += 1
        print(f"Downloading has been completed ({downloaded_reviews_count}/{total_reviews_count}). Yay!! ^-^")


if __name__ == '__main__':
    # app_id = 1245620    # Elden Ring
    
    app_ids = [1245620, 570, 292030, 570940, 236430, 374320, 367520, 435150, 1888160, 814380, 730, 588650]
    
    steam_works = SteamWorks()
    steam_works.download_reviews_to_csv_several_ids('reviews', app_ids)