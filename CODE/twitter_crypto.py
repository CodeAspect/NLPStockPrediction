
import requests
import os
import sys
import json
import time

class TwitterCrypto(object):
    """Notes:

        Altcoin Daily posts a lot of videos. Not seeing as much news."""

    # XXX These fields are not used, but could be useful
    # to someone using this code in the future

    world_event_accounts = ['DEFCONWSALERTS']
    crypto_keywords = set([
        '$btc', 'btc', 'bitcoin', 'bitcorn',
        '$eth', 'eth', 'ether', 'ethereum',
        '$ada', 'ada', 'cardano',
        '$matic', 'matic',
        '$xrp','xrp'])
    misc_terms = set(['hold', 'hodl', 'whale', 'happening', 'congress', 'sec'])
    good_terms = set(['ath', 'all-time high', 'high', 'buy', 'bull', 'approve', 'rally', 'jump', 'record', 'up'])
    negative_terms = set(['illegal', 'sanction', 'fud', 'sell', 'crash', 'bear', 'dip', 'low', 'down'])
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
    def create_account_url(self, username):
        # def get_users(self, usernames, user_fields):
        # Specify the usernames that you want to lookup below
        # You can enter up to 100 comma-separated values.
        usernames = "usernames={}".format(username)
        user_fields = "user.fields=description,id,created_at"
        # User fields are adjustable, options include:
        # created_at, description, entities, id, location, name,
        # pinned_tweet_id, profile_image_url, protected,
        # public_metrics, url, username, verified, and withheld
        url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
        return url

    def _bearer_oauth(self):
        BEARER_TOKEN = self.bearer_token
        def bearer_oauth(r):
            """
            Method required by bearer token authentication.
            """

            r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
            r.headers["User-Agent"] = "v2UserLookupPython"
            return r
        return bearer_oauth

    def create_tweets_url(self):
        search_url = "https://api.twitter.com/2/users//tweets/"
        #Change to the endpoint you want to collect data from
        #change params based on the endpoint you are using
        query_params = {
                    'exclude' : 'replies',
                    'max_results' : '',
		    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
		    'tweet.fields': 'id,text,author_id,created_at,public_metrics,',
		    'next_token': {}}
        return (search_url, query_params)

    def connect_to_endpoint(self, url):
        response = requests.request("GET", url, auth=self._bearer_oauth(),)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()

    def get_user_id(self, username):
        url = self.create_account_url(username)
        json = self.connect_to_endpoint(url)
        return json['data'][0]['id']

    def get_tweets(self, user_id):
        # url = self.create_tweets_url()
        next_token = None
        data = []
        while True:
            tweets = self.connect_to_endpoint2(user_id, next_token)
            if 'data' in tweets.keys():
                data.append(tweets['data'])
                if data == None:
                    break
                if 'meta' in tweets.keys():
                    meta = tweets['meta']
                    if 'next_token' in meta.keys():
                        next_token = meta['next_token']
                    else:
                        break
            else:
                print(tweets)
        return data

    def connect_to_endpoint2(self, user_id, next_token):
        """
        TODO: Add pagination_token: pagination_token
        """
        url = "https://api.twitter.com/2/users/{}/tweets".format(user_id) # self.create_tweets_url()
        params = {"tweet.fields": "created_at,public_metrics", "max_results" : 100} # self.get_params()
        if next_token:
            params['pagination_token'] = next_token
        response = requests.request("GET", url, auth=self._bearer_oauth(), params=params)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
        return response.json()



def usage():
    print("{} [bearer_token]".format(sys.argv[0]))

def main():

    if len(sys.argv) != 2:
        usage()
        exit(1)
    bearer_token = sys.argv[1]
    tc = TwitterCrypto(bearer_token)
    print("tc.bearer_token: ", tc.bearer_token)

    accounts = [
            ]
    crawled_accounts = [
            'cryptocom',
            'ForbesCrypto',
            'brian_armstrong',
            'BTC_Archive',
            'DocumentingBTC',
            'whale_alert',
            'crypto',
            'TrustWallet',
            'elliotrades',
            'aantonop',
            'danheld',
            'TheCryptoDog',
            'APompliano',
            'HsakaTrades',
            'bloodgoodBTC',
            'BTCTN',
            'BitcoinMagazine',
            'garyvee']

    amzn_accounts = []
    aapl_accounts = []
    gme_accounts = ['pulte', 'GMEshortsqueeze', '_tradespotting','WSBArmy', 'heyitspixel69']

    for account in accounts:
        time.sleep(60*2)
        print(account)
        with open(account + '.json', 'w') as f:
            user_id = tc.get_user_id(account)
            json_response = tc.get_tweets(user_id)
            f.write(json.dumps(json_response, indent=4, sort_keys=True))

if __name__ == '__main__':
    main()
