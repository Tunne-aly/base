from urllib.request import urlopen
from bs4 import BeautifulSoup
import sys
from time import time, sleep
from csv import writer
from re import search

base_url = ""
t = time()


def main(l):
    for url in l:
        do_stuff(url)


def do_stuff(url):
    pagenum = 1
    soup = getsoup("{}/{}".format(url, pagenum))
    r_pages = get_review_urls(soup)
    maxnum = get_num_pages(soup)
    extract_reviews(r_pages, url.split("/")[4])
    while pagenum < maxnum:
        pagenum += 1
        soup = getsoup("{}/{}".format(url, pagenum))
        r_pages = get_review_urls(soup)
        extract_reviews(r_pages, url.split("/")[4])


def extract_reviews(r_pages, fn):
    revs = []
    for page in r_pages:
        for rev in get_reviews(page):
            revs.append((rev[0], rev[1], page, rev[2], rev[3]))
    with open("{}.csv".format(fn), 'a') as f:
        w = writer(f)
        for rev in revs:
            w.writerow(rev)
        print("wrote {} reviews to file".format(len(revs)))


def get_num_pages(soup):
    txt = soup.find("span", {"class": "page-selector__page-count-indicator"}).text
    return int(search("\d+", txt).group(0))


def get_reviews(link):
    rev_list = []
    s = getsoup(link)
    revs = s.find_all("div", {"class": "reviews-page-review"})
    for r in revs:
        rev_list.append(get_review(r))
    return rev_list


def get_review(elem):
    rating = elem.find("span", {"class": "star-ratings__percentage"}).get("data-average-rating")
    review = elem.find("div", {"itemprop": "content__description"}).get_text()
    tu = elem.find_all("button", {"class": "thumb-button"})
    return rating, review, tu[0].get_text(), tu[1].get_text()


def getsoup(url):
    global t
    while time() - 2 <= t:
        sleep(1)
    while True:
        try:
            s = BeautifulSoup(urlopen("{}{}".format(base_url, url)).read(), "html.parser")
            break
        except:
            print("error retrieving {}, retrying in 3...")
            sleep(3)
    print("Retrieved {}".format(url))
    t = time()
    return s


def get_review_urls(soup):
    links = [a.get("href") for a in soup.find_all("a", {"class": "data__list-product-link"})]
    links = [make_review(l) for l in links]
    return links


def make_review(l):
    s = l.split("/")
    s[2] = "reviews"
    return "/".join(s[:-1])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("specify base url and at least one entry point.")
    else:
        base_url = sys.argv[1]
        main(sys.argv[2:])
