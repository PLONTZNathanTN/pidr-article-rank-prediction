import pandas as pd
from thefuzz import process  # For fuzzy matching

# --- Helper function: Extract the conference ID from a DBLP key ---
def extract_conf_id(dblp_key):
    """
    Extracts the conference ID from a DBLP key.
    For example, 'conf/models/2022' returns 'models'.

    Parameter:
        dblp_key (str): The DBLP key (e.g., 'conf/models/2022').
    Returns:
        str or None: The extracted ID, or None if not possible.
    """
    if not dblp_key:
        return None
    parts = dblp_key.split("/")
    if len(parts) >= 2:
        return parts[1].lower()  # Normalize to lowercase
    return None

# --- Function 2: Search in CORE.csv using the conference label and DBLP key ---
def get_conf_and_rank_by_conf(conf_label, key, csv_path="data/CORE2023.csv", threshold=80):
    """
    From a paper's conference label and DBLP key, retrieves the conference or journal
    and matches it to the CORE CSV file to get its ranking.

    The search first attempts to match the DBLP-extracted ID with an exact match.
    If no match is found or the ID is 'N/A', fuzzy matching is used on the full conference name or acronym.

    Parameters:
        conf_label (str): Full name of the conference.
        key (str): The DBLP key.
        csv_path (str): Path to the updated CORE CSV file (with "DBLP KEY" column).
        threshold (int): Fuzzy matching score threshold for a valid match.

    Returns:
        tuple (matched_identifier, rank, score) or (None, None, 0) if not found.
    """
    # 1. Load the updated CORE CSV file
    try:
        core_df = pd.read_csv(csv_path, sep=",", header=None, 
                              names=["ID", "Conference Name", "Acronym", "Source Year", "Rank", 
                                     "Included", "Code1", "Code2", "Code3", "DBLP LINK", "DBLP KEY"],
                              engine="python")
    except Exception as e:
        print("Error reading CORE.csv:", e)
        return None, None, 0

    # 2. Try to match using the extracted DBLP ID
    article_conf_id = extract_conf_id(key)
    if article_conf_id:
        # Look for an exact match on the DBLP KEY field
        for idx, row in core_df.iterrows():
            csv_dblp_key = row["DBLP KEY"]
            if csv_dblp_key and csv_dblp_key == article_conf_id:
                rank = row["Rank"]
                print(f"Exact match found by DBLP ID: {csv_dblp_key} with rank {rank}")
                return csv_dblp_key, rank, 100
    else:
        print("No DBLP ID extracted from the article key.")

    # 3. If no ID match is found, perform fuzzy matching on name or acronym
    conference_names = core_df["Conference Name"].dropna().tolist()
    acronym_names = core_df["Acronym"].dropna().tolist()
    best_match_conf_name, score_conf_name = process.extractOne(conf_label, conference_names)
    best_match_acronym, score_conf_acronym = process.extractOne(conf_label, acronym_names)
    if score_conf_name >= score_conf_acronym:
        if score_conf_name >= threshold:
            print(f"Best match by name: '{best_match_conf_name}' with score {score_conf_name}")
            rank = core_df.loc[core_df["Conference Name"] == best_match_conf_name, "Rank"].iloc[0]
            return best_match_conf_name, rank, score_conf_name
        else:
            print("No satisfactory match found in CORE.csv.")
            return None, None, 0
    else:
        if score_conf_acronym >= threshold:
            print(f"Best match by acronym: '{best_match_acronym}' with score {score_conf_acronym}")
            rank = core_df.loc[core_df["Acronym"] == best_match_acronym, "Rank"].iloc[0]
            return best_match_acronym, rank, score_conf_acronym
        else:
            print("No satisfactory match found in CORE.csv.")
            return None, None, 0

# --- Wrapper function to get best match (conference or journal) ---
def get_rank(label, key):
    """
    Tries to retrieve the rank and label (conference or journal) using the title and DBLP key.

    Parameters:
        label (str): Conference or journal label.
        key (str): DBLP key.
    
    Returns:
        tuple: (best_label, rank, score, type) or fallback values.
    """
    if not label and not key:
        return None, None, None, None
    
    best_conf, rank_conf, score_conf = get_conf_and_rank_by_conf(label, key)

    if score_conf == 0 :
        return "None", "UnRanked", 0, "None"
    else:
        return best_conf, rank_conf, score_conf, "conference"
