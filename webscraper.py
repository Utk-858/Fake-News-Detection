import feedparser
import urllib.request
import urllib.error
import urllib.parse
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import socket

# Updated List of RSS feeds
RAW_RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
    "https://www.reutersagency.com/feed/?best-topics=technology&post_type=best",
    "https://www.reutersagency.com/feed/?best-topics=business&post_type=best",
    "http://rss.cnn.com/rss/edition_world.rss",
    "http://rss.cnn.com/rss/edition_technology.rss",
    "https://feeds.npr.org/1004/rss.xml",
    "https://feeds.npr.org/1019/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/technology/rss",
    "https://news.google.com/rss/search?q=space%20exploration&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=artificial%20intelligence&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=global%20warming&hl=en-US&gl=US&ceid=US:en"
]

# Helper to safely fetch a feed
def fetch_feed(url):
    try:
        return feedparser.parse(url)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Helper to fetch full article text
def fetch_full_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to focus on the <article> section if available
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        # Filter: Only keep longer paragraphs
        good_paragraphs = [p.get_text() for p in paragraphs if len(p.get_text()) > 40]

        full_text = ' '.join(good_paragraphs)
        return full_text[:5000]  # limit to 5000 characters
    except Exception as e:
        print(f"Error fetching article text from {url}: {e}")
        return ""

# Storage
articles = []

# Fetch feeds
for feed_url in RAW_RSS_FEEDS:
    feed = fetch_feed(feed_url)
    if not feed or not hasattr(feed, 'entries'):
        continue

    for entry in feed.entries:
        title = entry.get('title', '').strip()
        link = entry.get('link', '').strip()
        published = entry.get('published', '').strip()

        if title and link:
            content = fetch_full_text(link)
            articles.append({
                'title': title,
                'url': link,
                'published': published,
                'content': content
            })

        if len(articles) >= 30000:
            break
    if len(articles) >= 30000:
        break

# Save to CSV
print(f"\n Saving {len(articles)} articles...")

df = pd.DataFrame(articles)
df.to_csv('scraped_articles.csv', index=False, encoding='utf-8')

print(" Scraping completed!")