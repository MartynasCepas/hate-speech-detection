import praw
import csv
import datetime
import time
import logging
from dotenv import load_dotenv
import os

load_dotenv()
# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ---------------------------
# Configuration
# ---------------------------
MAX_COMMENTS = 5000   # How many total comments to fetch
MAX_SUBMISSIONS = 10000  # How many submissions to check (higher = slower but more coverage)
COMMENT_TREE_EXPANSION = None  # 'None' fetches all comments; 0 = only top-level

# 1. Fill in your credentials from your Reddit app
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# 2. Create the Reddit instance
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# 3. Set up the subreddit
subreddit_name = "lietuva"
subreddit = reddit.subreddit(subreddit_name)

# CSV Output files
submissions_file = "lietuvos_submissions.csv"
comments_file = "lietuvos_comments.csv"

# Initialize data storage
collected_submissions = []
collected_comments = []
total_comments_fetched = 0

def utc_to_datetime(utc_timestamp):
    return datetime.datetime.utcfromtimestamp(utc_timestamp)

logging.info("Starting to scrape subreddit: r/%s", subreddit_name)
logging.info("Fetching up to %d comments (max submissions to check: %d)", MAX_COMMENTS, MAX_SUBMISSIONS)

submission_count = 0

# Open CSV files for writing
with open(submissions_file, mode='w', newline='', encoding='utf-8') as sub_file, \
     open(comments_file, mode='w', newline='', encoding='utf-8') as com_file:

    sub_writer = csv.writer(sub_file)
    com_writer = csv.writer(com_file)

    # Write headers for CSV files
    sub_writer.writerow(["id", "title", "created_utc", "created_dt", "author", "score", "num_comments", "permalink"])
    com_writer.writerow(["comment_id", "submission_id", "author", "body", "created_utc", "created_dt", "score", "permalink"])

    # 4. Crawl submissions sorted by newest (limit=MAX_SUBMISSIONS).
    for submission in subreddit.new(limit=MAX_SUBMISSIONS):
        submission_count += 1
        submission_created_dt = utc_to_datetime(submission.created_utc)

        # Log every 5th submission processed
        if submission_count % 5 == 0:
            logging.info("Processing submission #%s, ID=%s, created=%s",
                         submission_count, submission.id, submission_created_dt)

        # Gather submission data
        sub_writer.writerow([
            submission.id, submission.title, submission.created_utc, submission_created_dt,
            str(submission.author), submission.score, submission.num_comments, submission.permalink
        ])

        # 5. Load all comments for the submission (up to the given replace_more limit)
        try:
            submission.comments.replace_more(limit=COMMENT_TREE_EXPANSION)
        except Exception as e:
            logging.warning("Could not replace_more for submission ID=%s: %s", submission.id, e)
            continue

        # Flatten comment tree
        comments_list = submission.comments.list()
        logging.debug("Submission ID=%s has %d comments (after replace_more).",
                      submission.id, len(comments_list))

        for c in comments_list:
            if total_comments_fetched >= MAX_COMMENTS:
                logging.info("Reached MAX_COMMENTS limit (%d). Stopping early.", MAX_COMMENTS)
                break

            com_writer.writerow([
                c.id, submission.id, str(c.author), c.body, c.created_utc, utc_to_datetime(c.created_utc),
                c.score, c.permalink
            ])

            total_comments_fetched += 1

        logging.info("Fetched %d comments for submission ID=%s", len(comments_list), submission.id)

        # Stop early if we reached the max comments
        if total_comments_fetched >= MAX_COMMENTS:
            break

        # Sleep a bit to avoid hitting Reddit’s rate limits
        time.sleep(1)

logging.info("Finished scraping. Total submissions processed: %d", submission_count)
logging.info("Collected %d total comments", total_comments_fetched)
logging.info("Saved submissions to %s", submissions_file)
logging.info("Saved comments to %s", comments_file)
