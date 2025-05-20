import os
import sys
import time
from tqdm import tqdm
from pymongo.collection import Collection
from matching.match_dblp_core import get_rank

def process_dblp_core_matching(collection: Collection, batch_size=6000):
    """
    Process a batch of articles from the MongoDB collection to match DBLP conferences with CORE rankings.

    Args:
        collection (pymongo.collection.Collection): The MongoDB collection of articles.
        batch_size (int): Number of articles to process in a single run.
    """

    # Add the project root directory (where the DB folder is located) to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Retrieve articles that haven't been processed for DBLP-CORE matching
    articles_to_process = list(collection.find(
        {"dblp_core_linked": {"$exists": False}}, 
        {"_id": 1, "dblp_conference": 1, "dblp_key": 1}
    ).limit(batch_size))

    if not articles_to_process:
        print("No new articles to process. All have been checked for DBLP-CORE matching.")
        return

    time_start = time.time()
    time_estimates = []

    with tqdm(total=len(articles_to_process), desc="Matching Papers", unit="paper", leave=True, dynamic_ncols=True) as pbar:
        for i, article in enumerate(articles_to_process, 1):
            conference = article["dblp_conference"]
            key = article["dblp_key"]

            if isinstance(conference, str):
                start_time = time.time()

                # Matching function
                conference_match, rank, score_rank, pub_type = get_rank(conference, key)

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
                        "dblp_core_linked": True,
                        "core_publication": conference_match,
                        "core_publication_score": score_rank,
                        "core_publication_rank": rank,
                        "publication_type": pub_type
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
