# app/crawler/crawler.py

import feedparser
import hashlib
from pymongo import MongoClient
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# --------- RSS feeds ---------
FEEDS = {
    "Newsbeast": "https://www.newsbeast.gr/feed",
    "Naftemporiki": "https://www.naftemporiki.gr/feed/rss",
    "Greek Reporter": "https://greekreporter.com/feed/",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
    "TechCrunch": "http://feeds.feedburner.com/TechCrunch/"
}

# --------- Θεματικές Κατηγορίες ---------
CATEGORIES = {
    "Politics": ["government", "election", "minister", "policy", "βουλή", "πολιτική"],
    "Economy": ["inflation", "market", "stock", "επενδύσεις", "οικονομία", "τράπεζα"],
    "Technology": ["ai", "artificial intelligence", "tech", "blockchain", "software", "τεχνολογία"],
    "Health": ["health", "covid", "disease", "hospital", "υγεία", "ιός", "εμβόλιο"],
    "World": ["conflict", "war", "UN", "Israel", "Russia", "world", "παγκόσμιος"],
    "Sports": ["match", "team", "goal", "league", "πρωτάθλημα", "ομάδα", "ποδόσφαιρο"],
}

# --------- MongoDB setup ---------
client = MongoClient("mongodb://172.25.240.1:27017/")
db = client["news_database"]
collection = db["articles"]

def classify_category(text):
    text = text.lower()
    for category, keywords in CATEGORIES.items():
        if any(keyword.lower() in text for keyword in keywords):
            return category
    return "General"

def compute_md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def crawl():
    print(" Ξεκίνησε το crawling ειδήσεων...")
    articles = []

    for name, feed_url in FEEDS.items():
        print(f" Crawling {name}...")
        try:
            feed = feedparser.parse(feed_url)
            fetched_articles = []

            for entry in feed.entries[:15]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                published_date = entry.get("published", datetime.now().isoformat())
                content = entry.get("summary", "") or entry.get("description", "")
                content = content.strip()

                # Ποιότητα άρθρου
                if not title or not link or len(content) < 300:
                    continue

                category = classify_category(title + " " + content)
                content_hash = compute_md5(content)

                fetched_articles.append({
                    "title": title,
                    "link": link,
                    "published_date": published_date,
                    "content": content,
                    "category": category,
                    "hash": content_hash
                })

            articles.extend(fetched_articles)
            print(f" Crawled {len(fetched_articles)} articles from {name}\n")

        except Exception as e:
            print(f" Error crawling {name}: {e}\n")

    if articles:
        insert_articles_to_mongo(articles)
        print(f" Ολοκληρώθηκε το crawling! Συνολικά άρθρα μαζεύτηκαν: {len(articles)}\n")
    else:
        print(" Δεν βρέθηκαν άρθρα!\n")

def insert_articles_to_mongo(articles):
    for article in articles:
        existing = collection.find_one({"link": article["link"]})
        if not existing:
            collection.insert_one(article)
        elif existing.get("hash") != article["hash"]:
            collection.update_one({"_id": existing["_id"]}, {"$set": article})
            print(f"🔁 Ενημερώθηκε άρθρο: {article['link']}")
        else:
            print(f"⚠️ Ήδη υπάρχει χωρίς αλλαγή: {article['link']}")

def scheduled_crawl():
    crawl()
    scheduler = BackgroundScheduler()
    scheduler.add_job(crawl, "interval", hours=1)
    scheduler.start()
    print(" Προγραμματίστηκε crawling κάθε 1 ώρα!\n")
