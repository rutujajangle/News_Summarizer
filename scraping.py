import nltk
from newspaper import Article

nltk.download("punkt")


def scrape(url):
    article = Article(url)
    article.download()
    article.parse()

    return article.text


if __name__ == "__main__":
    print(1)
    print(nltk.__version__)
    print(
        scrape(
            "https://www.ndtv.com/world-news/us-presses-russia-on-ukraine-in-highest-level-one-on-one-meet-since-war-3828668"
        )
    )
