from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import datetime, time, requests, json
from bs4 import BeautifulSoup
try:
    limit = datetime.datetime.strptime("2019-02-02", "%Y-%m-%d")

    driver = webdriver.Chrome('./chromedriver')
    driver.get("https://www.reuters.com/companies/GME.N/news")

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(1)

    articles = driver.find_elements(by=By.CLASS_NAME, value="item")
    count = len(articles)

    prevCount = 0
    while True:
        #lastEleDate = articles[-1].find_element(by=By.TAG_NAME, value="time").text

        if count == prevCount:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            break
        else:
            prevCount = count

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)

        articles = driver.find_elements(by=By.CLASS_NAME, value="item")
        count = len(articles)

    _json = []
    for article in reversed(articles):
        tag = article.find_element(by=By.TAG_NAME, value="a")
        url = tag.get_attribute('href')

        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        data = soup.find("div", {"class","article__container__2MUeZ"})

        try:
            if data != None:
                date = data.find("time").findChildren()
                temp = datetime.datetime.strptime(str(date[0].text), '%B %d, %Y')
                date = temp.strftime('%Y-%m-%d')

                if(datetime.datetime.strptime(date, "%Y-%m-%d") >= limit):
                    title = data.find("div", {"class","article-header__heading__15OpQ"}).text

                    body = data.find("div", {"class","article-body__content__3VtU3 paywall-article"}).text

                    _json.append([str(title), date, str(body)])
            else:
                newDriver = webdriver.Chrome('./chromedriver')
                newDriver.get(url)

                date = newDriver.find_element(by=By.CLASS_NAME, value="ArticleHeader-date-line-3oc3Y")
                date = date.find_element(by=By.TAG_NAME, value="time")
                
                temp = datetime.datetime.strptime(str(date.text), '%B %d, %Y')
                date = temp.strftime('%Y-%m-%d')

                if(datetime.datetime.strptime(date, "%Y-%m-%d") >= limit):
                    title = soup.find("div", {"class","ArticlePage-article-header-23J2O"}).find('h1').text

                    body = soup.find("div", {"class","ArticleBodyWrapper"}).find_all("p")
                    bodyText = ''
                    for b in body:
                        bodyText = bodyText + str(b.text)

                    _json.append([str(title), date, bodyText])
                newDriver.quit()
                
        except:
            print("Error with article: "+str(url))
            newDriver.quit()
finally:

    driver.quit()

    _json = json.dumps(_json)

    f = open('GMEnews.txt', 'w')
    f.write(_json)
    f.close()
