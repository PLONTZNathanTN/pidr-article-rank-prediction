import requests
import re
import time
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from thefuzz import process
from thefuzz import fuzz

DBLP_API_URL = "https://dblp.uni-trier.de/search/publ/api"
DBLP_SPARQL_URL = "https://sparql.dblp.org/sparql"

def search_dblp_by_title(arxiv_title, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(f"{DBLP_API_URL}?q={arxiv_title}&format=json", timeout=10)
            if response.status_code == 429:
                wait_time = 5 + retries * 2
                tqdm.write(f"Rate limited! Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
                continue
            response.raise_for_status()
            data = response.json()

            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                return False, None, None, None, 0, "No good match"

            candidates = [
                (
                    hit["info"]["title"],
                    hit["info"].get("venue", "No Conference Found"),
                    hit["info"].get("key", "No Key Found")
                )
                for hit in hits
            ]

            best_match, score = process.extractOne(arxiv_title, [t[0] for t in candidates])

            if score < 90:
                #print(f"No good title match for: {arxiv_title} (Best match: {best_match} with score {score})")
                return False, None, None, None, score, "No good match"

            best_conf, best_key = next((conf, key) for title, conf, key in candidates if title == best_match)
            return True, best_match, best_conf, best_key, score, "API title search"

        except requests.RequestException as e:
            #print(f"Error querying DBLP for title: {arxiv_title} - {e}")
            return False, None, None, None, 0, "No good match"
    
    return False, None, None, None, 0, "No good match"


def search_dblp_by_authors(arxiv_title, authors, max_retries=10):
    if not authors:
        return False, None, None, None, 0, "No good match"
    
    author_query = " ".join(authors)
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(f"{DBLP_API_URL}?q=author:{author_query}&format=json", timeout=10)
            if response.status_code == 429:
                wait_time = 5 + retries * 2
                tqdm.write(f"Rate limited! Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
                continue
            response.raise_for_status()
            data = response.json()

            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                #print(f"No results found in DBLP for authors: {author_query}")
                return False, None, None, None, 0, "No good match"

            candidates = [
                (
                    hit["info"]["title"],
                    hit["info"].get("venue", "No Conference Found"),
                    hit["info"].get("key", "No Key Found")
                )
                for hit in hits
            ]

            best_match, score = process.extractOne(arxiv_title, [t[0] for t in candidates],scorer=fuzz.ratio)
            
            

            if score < 70:
                #print(f"No good author match for: {author_query} (Best match: {best_match} with score {score})")
                return False, None, None, None, score, "No good match"

            best_conf, best_key = next((conf, key) for title, conf, key in candidates if title == best_match)
            return True, best_match, best_conf, best_key, score, "API author search"

        except requests.RequestException as e:
            #print(f"Error querying DBLP for authors: {author_query} - {e}")
            return False, None, None, None, 0, "No good match"
    
    return False, None, None, None, 0, "No good match"


def search_dblp_by_sparql(arxiv_title):
    """
    Searches DBLP using SPARQL for the closest paper title match and retrieves the associated conference and DBLP key.

    Parameters:
        arxiv_title (str): The title from ArXiv.

    Returns:
        tuple (bool, str, str, str, int, str): 
            - Matched flag
            - Best-matching title
            - Conference name
            - DBLP key
            - Fuzzy match confidence score
            - Search method used
    """
    sparql = SPARQLWrapper(DBLP_SPARQL_URL)
    
    # Clean the title by removing common prefixes
    clean_title = re.sub(r'^(Abstract|Introduction|Summary|Overview|Conclusion):\s*', '', arxiv_title, flags=re.IGNORECASE)

    # Extract meaningful words (4+ characters) for better matching
    extracted_words = re.findall(r'\b\w{4,}\b', clean_title)
    title_keywords = f".*{'.*'.join(extracted_words)}.*" if extracted_words else re.escape(clean_title)

    query = f"""
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?title ?confName ?article
    WHERE {{
      ?article dblp:title ?title .
      OPTIONAL {{
        ?article dblp:publishedInStream ?stream .
        ?stream rdfs:label ?confName .
      }}
      FILTER(REGEX(LCASE(?title), "{title_keywords}", "i"))
    }}
    LIMIT 10
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if not bindings:
            return False, None, None, None, 0, "No good SPARQL match"

        # Extract potential candidates from the results
        candidates = [
            (
                entry["title"]["value"],
                entry.get("confName", {}).get("value", "No Conference Found"),
                entry["article"]["value"].split("/")[-1]
            ) 
            for entry in bindings
        ]

        # Apply fuzzy matching
        best_match, score = process.extractOne(arxiv_title, [t[0] for t in candidates])

        # Reject matches with a score below 90
        if score < 90:
            #print(f"No good SPARQL match for: {arxiv_title} (Best match: {best_match} with score {score})")
            return False, None, None, None, score, "No good match"

        # Retrieve conference name and key for the best match
        best_conf, best_key = next((conf, key) for title, conf, key in candidates if title == best_match)

        return True, best_match, best_conf, best_key, score, "SPARQL search"

    except Exception as e:
        #print(f"Error querying DBLP SPARQL for title: {arxiv_title} - {e}")
        return False, None, None, None, 0, "No good match"


def match_arxiv_with_dblp(arxiv_title, authors, deepness=3):
    """
    Attempts to match an ArXiv paper to a DBLP paper using a multi-step fallback mechanism.

    Parameters:
        arxiv_title (str): The title of the paper from ArXiv.
        authors (list): A list of author names from the ArXiv paper.
        deepness (int): Controls the depth of the search.
            - If 1, only searches by title using the DBLP API.
            - If 2, searches by title and falls back to a DBLP SPARQL query.
            - If 3 or more, searches by title, author, and finally falls back to a DBLP SPARQL query.

    Returns:
        tuple:
        - matched (bool): True if a match is found, False otherwise.
        - best_title (str or None): The best-matching title from DBLP.
        - best_conf (str or None): The associated conference name.
        - best_key (str or None): The DBLP key of the matched paper.
        - score (int): The fuzzy matching confidence score.
        - method (str): The method used for the match ("API title search", "API author search", or "SPARQL search").
    """
    
    if deepness == 1:
        return search_dblp_by_title(arxiv_title)
    elif deepness == 2:
        matched, best_title, best_conf, best_key, score, method = search_dblp_by_title(arxiv_title)
        if not matched:
            return search_dblp_by_sparql(arxiv_title)
        return matched, best_title, best_conf, best_key, score, method
    else:
        matched, best_title, best_conf, best_key, score, method = search_dblp_by_title(arxiv_title)
        if not matched:
            matched, best_title, best_conf, best_key, score, method = search_dblp_by_sparql(arxiv_title)
            if not matched:
                return search_dblp_by_authors(arxiv_title,authors)
        return matched, best_title, best_conf, best_key, score, method


        
    


    
