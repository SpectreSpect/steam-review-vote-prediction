import requests
import pandas as pd


class SteamWorks:
    def __init__(self):
        pass

    def get_reviews_and_voted_ups(self, app_id):
        num_per_page = 100
        cursor = '*'
        url = f'https://store.steampowered.com/appreviews/{app_id}?json=1&num_per_page={num_per_page}&language=all&cursor={cursor}'
        response = requests.get(url)
        total_reviews = response.json()['query_summary']['total_reviews']
        current_total_reviews = 0
        rows = []
        while True:
            response = requests.get(f'https://store.steampowered.com/appreviews/{app_id}?json=1&'
                                    f'num_per_page={num_per_page}&'
                                    f'language=all&'
                                    f'cursor={cursor}')
            response_json = response.json()
            success = response_json['success']
            if success == 1:
                cursor = response_json['cursor']
                num_reviews = response_json['query_summary']['num_reviews']
                current_total_reviews += num_reviews
                print(f"{current_total_reviews}/{total_reviews}")

                for j in range(num_reviews):
                    recommendation_id = response_json['reviews'][j]['recommendationid']
                    review = response_json['reviews'][j]['review']
                    voted_up = response_json['reviews'][j]['voted_up']
                    new_row = {'recommendationid': recommendation_id, 'review': review, 'voted_up': voted_up}
                    rows.append(new_row)
            else:
                print('Request failed')
                print(f"error: {response_json['error']}")
                break
        data_frame = pd.DataFrame(rows, columns=['recommendationid', 'review', 'voted_up'])
        return data_frame


if __name__ == '__main__':
    steam_works = SteamWorks()
    data_frame = steam_works.get_reviews_and_voted_ups(1245620)


    # # Elden ring: 1245620
    # # Random game: 2383990
    #
    # app_id = 2383990
    # num_per_page = 20  # from 20 to 100
    # cursor = '*'
    #
    # url = f'https://store.steampowered.com/appreviews/{app_id}?json=1&num_per_page={num_per_page}&language=all&cursor={cursor}'
    #
    # r = requests.get(url)
    # response_json = r.json()
    # total_reviews = response_json['query_summary']['total_reviews']
    #
    # print(f'url: {url}')
    # print(f"Total reviews: {total_reviews}")
    #
    # r = 0
    # for i in range(20):
    #     r = requests.get(f'https://store.steampowered.com/appreviews/{app_id}?'
    #                      f'json=1&'
    #                      f'num_per_page={num_per_page}&'
    #                      f'cursor={cursor}')
    #     response_json = r.json()
    #     success = response_json['success']
    #     if success == 1:
    #         cursor = response_json['cursor']
    #         num_reviews = response_json['query_summary']['num_reviews']
    #         print('-' * 20 + f'i: {i}' + '-' * 20)
    #         print(f"Cursor: {cursor}")
    #         print(f"Num reviews: {num_reviews}")
    #         for j in range(num_reviews):
    #             print(response_json['reviews'][j]['recommendationid'])
    #     else:
    #         print('Request failed')
    #         print(f"error: {response_json['error']}")
    #         break
    #
    #
    # # r = requests.get(f'https://store.steampowered.com/appreviews/1245620?json=1&cursor={cursor}')
    # # cursor = r.json()['cursor']
    # # print(r.json()['reviews'][0]['review'])
    # #
    # # r = requests.get(f'https://store.steampowered.com/appreviews/1245620?json=1&cursor={cursor}')
    # # cursor = r.json()['cursor']
    # # print(r.json()['reviews'][0]['review'])
    #
    # # print('Recommendation id:')
    # # print(f"    {r.json()['reviews'][0]['recommendationid']}")
    # # print('Review:')
    # # print(f"    {r.json()['reviews'][0]['review']}")
    # # print('Vote:')
    # # print(f"    {r.json()['reviews'][0]['voted_up']}")
