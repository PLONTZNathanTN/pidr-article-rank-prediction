import os
import sys
import time
from tqdm import tqdm
from matching.match_arxiv_dblp import match_arxiv_with_dblp
from pymongo.collection import Collection

def process_arxiv_dblp_matching(collection: Collection, batch_size=6000):
    """
    Process a batch of articles from the MongoDB collection to match ArXiv papers with DBLP entries.

    Args:
        collection (pymongo.collection.Collection): The MongoDB collection of articles.
        batch_size (int): Number of articles to process in a single run.
    """

    # Add the project root directory (where the DB folder is located) to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Retrieve articles that haven't been matched yet
    articles_to_process = list(collection.find(
        {"arxiv_dblp_linked": {"$exists": False}}, 
        {"_id": 1, "title": 1, "authors": 1}
    ).limit(batch_size))

    if not articles_to_process:
        print("No new articles to process. All have been checked for ArXiv-DBLP matching.")
        return

    time_start = time.time()
    time_estimates = []

    with tqdm(total=len(articles_to_process), desc="Matching Papers", unit="paper", leave=True, dynamic_ncols=True) as pbar:
        for i, article in enumerate(articles_to_process, 1):
            title = article["title"]
            authors = article.get("authors", [])

            start_time = time.time()

            # Matching function
            is_match, dblp_title, conference, key, score, method = match_arxiv_with_dblp(title, authors, deepness=3)

            elapsed_time = time.time() - start_time
            time_estimates.append(elapsed_time)

            # Estimate remaining time
            avg_time_per_paper = sum(time_estimates) / len(time_estimates)
            remaining_papers = len(articles_to_process) - i
            estimated_remaining_time = remaining_papers * avg_time_per_paper

            # Update MongoDB with the matching result
            collection.update_one(
                {"_id": article["_id"]},
                {"$set": {
                    "arxiv_dblp_linked": True,
                    "matched": is_match, 
                    "dblp_title": dblp_title,
                    "dblp_conference": conference,
                    "dblp_key": key,
                    "title_matching_score": score,
                    "matching_method": method
                }}
            )

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "Estimated Remaining Time": time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))
            })

    time_end = time.time()
    print(f"\nSuccessfully processed {len(articles_to_process)} articles.")
    print(f"Total processing time: {time_end - time_start:.2f} seconds.")
