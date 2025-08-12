import argparse
import os
import pandas as pd
import datetime
from datetime import timedelta, timezone
import praw
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set up argument parser with defaults
parser = argparse.ArgumentParser(description="Fetch Reddit comments for sentiment analysis")
parser.add_argument("--subreddit", default="python", help="Subreddit name (default: python)")
parser.add_argument("--days", type=int, default=7, help="Number of days back to fetch (default: 7)")
parser.add_argument("--post_limit", type=int, default=100, help="Number of posts to fetch (default: 100)")
parser.add_argument("--out", default="data/raw_comments.csv", help="Output CSV path (default: data/raw_comments.csv)")
args = parser.parse_args()

# Make sure data folder exists
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# Set up Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Calculate 'after' timestamp
after_ts = int((datetime.datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp())

print(f"Fetching from r/{args.subreddit} for the last {args.days} days...")

# Fetch posts & comments
all_comments = []
for submission in reddit.subreddit(args.subreddit).new(limit=args.post_limit):
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        all_comments.append({
            "id": comment.id,
            "post_id": submission.id,
            "author": str(comment.author),
            "body": comment.body,
            "created_utc": datetime.datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
            "score": comment.score
        })

# Convert to DataFrame
df = pd.DataFrame(all_comments)

if df.empty:
    print("No comments found.")
else:
    # Save to CSV
    df.to_csv(args.out, index=False)
    print(f"✅ Saved {len(df)} comments to {args.out}")
    print(df.head())
