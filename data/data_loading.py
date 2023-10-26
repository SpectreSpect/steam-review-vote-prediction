import requests
import pandas as pd
import keyboard
import random
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
    def get(app_id, num_per_page=100, language='english', filter_='updated', cursor='*', **kwargs):
        url = f"https://store.steampowered.com/appreviews/{app_id}?" \
               f"json=1&" \
               f"filter={filter_}&" \
               f"num_per_page={num_per_page}&" \
               f"language={language}&" \
               f"cursor={cursor}"
        # print(url)
        return requests.get(url)

    def stop_loading_reviews(self, e):
        self.reviews_loading_force_stop = True
        print('')

    def stop_downloading_reviews_to_csv(self, e):
        self.reviews_downloading_to_csv_force_stop = True
        print('')

    def get_reviews_and_voted_ups(self, app_id, cursor='*',
                                  max_reviews_in_memory=10000,
                                  downloaded_reviews_count=0, **kwargs):
        cursors = []
        response = SteamWorks.get(app_id)
        total_reviews = response.json()['query_summary']['total_reviews']
        current_total_reviews = 0
        rows = []

        keyboard.on_press_key("c", self.stop_loading_reviews)

        while True:
            if self.reviews_loading_force_stop or self.reviews_downloading_to_csv_force_stop:
                print('Loading was forcefully stopped')
                self.reviews_loading_force_stop = False
                break

            if current_total_reviews >= max_reviews_in_memory:
                print(f'Maximum number of reviews ({max_reviews_in_memory}) in memory has been exceeded.')
                break

            response = SteamWorks.get(app_id, cursor=to_percent_encoding(cursor))
            response_json = response.json()
            success = response_json['success']
            if success == 1:
                cursor = response_json['cursor']

                if cursor in cursors:
                    print(f'Found a duplicate cursor: {cursor}')
                    break
                cursors.append(cursor)

                num_reviews = response_json['query_summary']['num_reviews']
                current_total_reviews += num_reviews
                downloaded_reviews_count += num_reviews
                print(f'    downloaded: {downloaded_reviews_count}/{total_reviews}, '
                      f'reviews_in_memory: {current_total_reviews}, '
                      f'cursor: {cursor}')
                for j in range(num_reviews):
                    recommendation_id = response_json['reviews'][j]['recommendationid']
                    review = response_json['reviews'][j]['review']
                    voted_up = response_json['reviews'][j]['voted_up']
                    new_row = {'app_id': app_id, 'recommendationid': recommendation_id,
                               'review': review, 'voted_up': voted_up}
                    rows.append(new_row)

                if current_total_reviews >= total_reviews:
                    break
            else:
                print('Request failed')
                print(f"error: {response_json['error']}")
                break

        data_frame = pd.DataFrame(rows, columns=['app_id', 'recommendationid', 'review', 'voted_up'])
        return {'data_frame': data_frame,
                'last_cursor': cursors[-1],
                'downloaded_reviews_count': current_total_reviews,
                'total_reviews': total_reviews}

    def download_reviews_to_csv(self, path, app_id, cursor="*", max_reviews_in_memory=5000, **kwargs):
        file_id = random.randint(0, 10 ** 10)

        keyboard.on_press_key("q", self.stop_downloading_reviews_to_csv)
        last_cursor = cursor
        total_downloaded_reviews_count = 0
        total_reviews = -1
        total_elapsed_time = 0
        i = 0
        while True:
            if self.reviews_downloading_to_csv_force_stop:
                self.reviews_downloading_to_csv_force_stop = False
                print('Downloading reviews to csv was forcefully stopped.')
                break



            start = timer()
            reviews_data = self.get_reviews_and_voted_ups(app_id,
                                                          cursor=last_cursor,
                                                          max_reviews_in_memory=max_reviews_in_memory,
                                                          downloaded_reviews_count=total_downloaded_reviews_count)
            end = timer()
            elapsed_time = end - start
            total_elapsed_time += elapsed_time
            data_frame = reviews_data['data_frame']
            total_downloaded_reviews_count += reviews_data['downloaded_reviews_count']
            total_reviews = reviews_data['total_reviews']
            last_cursor = reviews_data['last_cursor']
            print(f'Saving data... ')
            print(f"Reviews downloaded: {total_downloaded_reviews_count}/{total_reviews}, "
                  f"last_cursor: {last_cursor}, "
                  f"elapsed_time: {elapsed_time}, "
                  f"total_elapsed_time: {total_elapsed_time}")
            if i <= 0:
                data_frame.to_csv(f'{path}/app_id{app_id}file_id{file_id}.csv', index=False)
            else:
                data_frame.to_csv(f'{path}/app_id{app_id}file_id{file_id}.csv', index=False, mode='a', header=False)
            del data_frame

            if total_downloaded_reviews_count >= total_reviews:
                print(f"Downloading has been completed ({total_downloaded_reviews_count}/{total_reviews}). Yay!! ^-^")
                break

            i += 1
        print(f"Reviews downloaded: {total_downloaded_reviews_count}/{total_reviews}, "
              f"Last cursor: {last_cursor}")


if __name__ == '__main__':
    app_id = 1245620    # Elden Ring
    steam_works = SteamWorks()
    steam_works.download_reviews_to_csv('reviews', app_id)
