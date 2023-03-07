#!/usr/bin/env python3
import requests
import json
from bs4 import BeautifulSoup

def crawl_pages(results):
    data = []
    for result in results:
        publicationDate = result['webPublicationDate']
        title = result['webTitle']
        resp = requests.get(result['webUrl'])
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.body.get_text().strip()
        data.append({'date': publicationDate, 'title' : title, 'text' : text})
    return data

def get_guardian_news(api_key=None):
    """ """

    url = "https://content.guardianapis.com/search?page={}&q=(AMC OR AMZN OR BTC OR BITCOIN OR AMAZON OR AAPL OR APPLE)&from-date=2010-01-01&to-date=2023-03-30&api-key={}"
    print(url.format(1, api_key))
    resp = requests.get(url.format(1, api_key))
    print(resp)
    print(resp.json())
    print(resp.json())
    print(resp.json()['response'])
    pages = resp.json()['response']['pages']
    results = resp.json()['response']['results']
    data = crawl_pages(results)

    with open("news/news-guardian-{}.json".format(1), "w") as f:
        json.dump(data, f)

    if resp.status_code != 200:
        print("Error returning data")
        exit(-1)
    for i in range(2, pages):
        resp = requests.get(url.format(i, api_key))
        if resp.status_code != 200:
            print("Error returning data")
            exit(-1)
        results = resp.json()['response']['results']
        data = crawl_pages(results)
        with open("news/news-guardian-{}.json".format(i), "w") as f:
            json.dump(data, f)

def main():
    """ """
    import sys
    if len(sys.argv) != 2:
        print("{} <GUARDIAN API-KEY>".format(sys.argv[0]))
    get_guardian_news(api_key=sys.argv[1])

if __name__ == '__main__':
    main()
