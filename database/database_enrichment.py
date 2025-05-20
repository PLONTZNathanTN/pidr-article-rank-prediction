import requests
from lxml import etree
import xml.etree.ElementTree as ET
import json
import time
from pymongo.mongo_client import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
import textstat
import pandas as pd
from bs4 import BeautifulSoup
import os
from pdf_processing.pdf_windows_arxiv import download_arxiv_pdf
import random
import PyPDF2
import fitz  # PyMuPDF
import re
import sys
import codecs
from scholarly import scholarly
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection

def get_articles_collection(
    username: str,
    password: str
) -> Collection:
    """
    Connects to the MongoDB Atlas cluster and returns the 'Articles' collection
    from the 'ArticleDB' database using the provided credentials.

    Args:
        username (str): MongoDB username.
        password (str): MongoDB password.

    Returns:
        pymongo.collection.Collection: The 'Articles' collection.
    """
    cluster = "pidr.jtdn1.mongodb.net"
    database_name = "ArticleDB"
    collection_name = "Articles"
    app_name = "PIDR"

    uri = (
        f"mongodb+srv://{username}:{password}"
        f"@{cluster}/?retryWrites=true&w=majority&appName={app_name}"
    )
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[database_name]
    return db[collection_name]


def get_article_count_by_year(year):
    """
    Fetches the total number of cs.AI articles submitted in a specific year 
    using the arXiv API.

    Args:
        year (int): The year for which to count the articles.

    Returns:
        int or None: The total number of articles if successful, None otherwise.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:cs.AI AND submittedDate:[{year}01010000 TO {year}12312359]",
        "start": 0,
        "max_results": 1  # We only need the total count
    }

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        print("Error", response.status_code, "while requesting data from arXiv.")
        return None

    root = ET.fromstring(response.content)
    total_results = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')

    if total_results is None:
        print("Error: totalResults tag not found in response.")
        return None

    return int(total_results.text)

def fetch_articles_by_years():
    """
    Fetches a sample (10%) of cs.AI articles from arXiv for years 2018 to 2023.
    The articles are stored in a local JSON file called 'articles.json'.
    """
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    base_url = "http://export.arxiv.org/api/query"
    all_articles = []

    for year in years:
        total_articles = get_article_count_by_year(year)
        if not total_articles:
            continue

        sample_size = max(1, int(0.1 * total_articles))
        print(f"Fetching first {sample_size} articles for {year}...")

        articles_fetched = 0
        start = 0

        while articles_fetched < sample_size:
            remaining = sample_size - articles_fetched
            max_results = min(1000, remaining)

            params = {
                "search_query": f"cat:cs.AI AND submittedDate:[{year}01010000 TO {year}12312359]",
                "start": start,
                "max_results": max_results
            }

            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print("Error", response.status_code, "while requesting data for", year)
                break

            root = ET.fromstring(response.content)
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            articles = []
            for entry in root.findall('atom:entry', ns):
                doi_element = entry.find('arxiv:doi', ns)
                article = {
                    "title": entry.find('atom:title', ns).text,
                    "summary": entry.find('atom:summary', ns).text,
                    "published": entry.find('atom:published', ns).text,
                    "updated": entry.find('atom:updated', ns).text,
                    "id": entry.find('atom:id', ns).text,
                    "link": entry.find('atom:link', ns).attrib['href'],
                    "doi": doi_element.text if doi_element is not None else None,
                    "authors": [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                    "categories": [category.attrib['term'] for category in entry.findall('atom:category', ns)],
                    "comments": entry.find('arxiv:comment', ns).text if entry.find('arxiv:comment', ns) is not None else None,
                    "journal_ref": entry.find('arxiv:journal_ref', ns).text if entry.find('arxiv:journal_ref', ns) is not None else None,
                    "primary_category": entry.find('arxiv:primary_category', ns).attrib['term'] if entry.find('arxiv:primary_category', ns) is not None else None
                }
                articles.append(article)

            retrieved = len(articles)
            all_articles.extend(articles)
            articles_fetched += retrieved
            start += retrieved

            print(f"{articles_fetched}/{sample_size} articles fetched for {year}")

            time.sleep(2)

    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=4, ensure_ascii=False)

    print("Data saved to 'articles.json'")


def add_pdf_path(collection):
    """
    Adds a 'PDFPath' field to every document in the given collection,
    constructed as 'data/pdf/{arxiv_id}.pdf' based on the article's 'id' field.
    
    Args:
        collection (pymongo.collection.Collection): MongoDB collection.
    """
    for doc in collection.find({}, {"id": 1}):
        if "id" in doc:
            arxiv_id = doc["id"].split("/")[-1]  # Extract Arxiv identifier
            pdf_path = f"data/pdf/{arxiv_id}.pdf"
            collection.update_one({"_id": doc["_id"]}, {"$set": {"PDFPath": pdf_path}})

    print("Update completed.")



def create_pdf_folder_and_download(collection):
    """
    Fetches all article IDs from the given MongoDB collection,
    then downloads their PDFs using download_arxiv_pdf().

    Args:
        collection (pymongo.collection.Collection): MongoDB collection.

    Returns:
        None
    """
    try:
        # Retrieve all documents, only get the 'id' field
        articles = collection.find({}, {"id": 1, "_id": 0})

        ids = [article["id"] for article in articles if "id" in article]

        downloaded_count = 0
        processed_count = 0
        total_articles = len(ids)

        for arxiv_id in ids:
            # Assuming download_arxiv_pdf returns 2 on success for this specific case
            if download_arxiv_pdf(arxiv_id, downloaded_count, processed_count) == 2:
                processed_count += 1
            else:
                downloaded_count += 1

            processed_count += 1
            # Optional: time.sleep(5)

            print(f"PDFs downloaded: {processed_count}/{total_articles}")

        print("All PDFs have been downloaded successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")

        
def extract_pdf_info(file_path):
    """
    Extracts the number of pages and total character count from a PDF file.
    Also verifies the PDF can be opened with fitz (PyMuPDF).

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        tuple: (num_pages (int), num_characters (int)) if successful,
               (None, None) if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            num_characters = 0

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    num_characters += len(text)

        # Verify PDF can be opened by fitz (PyMuPDF)
        doc = fitz.open(file_path)
        doc.close()

        return num_pages, num_characters

    except Exception as e:
        print(f"Error extracting PDF info from {file_path}: {e}")
        return None, None
    

def update_pdf_info_in_db(collection):
    """
    Reads documents with a 'PDFPath' field from the given MongoDB collection,
    extracts PDF info (number of pages and characters),
    and updates each document with these values.

    Args:
        collection (pymongo.collection.Collection): MongoDB collection.
    """
    try:
        articles = collection.find({"PDFPath": {"$exists": True}})

        for article in articles:
            file_path = article.get("PDFPath")

            if file_path:
                num_pages, num_characters = extract_pdf_info(file_path)

                if num_pages is not None and num_characters is not None:
                    collection.update_one(
                        {"id": article["id"]},
                        {"$set": {"page": num_pages, "character": num_characters}}
                    )
                    print(f"Article {article['id']} updated successfully.")

    except Exception as e:
        print(f"Error updating database: {e}")



def extract_dblp_conference_id(link):
    """
    Extracts the conference ID from a DBLP URL.

    Examples:
        - 'https://dblp.uni-trier.de/db/conf/ssd'       --> 'ssd'
        - 'https://dblp.org/db/conf/nordichi'           --> 'nordichi'
        - 'N/A'                                         --> None
        - 'https://dblp.org/db/conf/wisec/index.html'   --> 'wisec'

    Args:
        link (str): The DBLP link string.

    Returns:
        str or None: Extracted conference ID in lowercase, or None if invalid.
    """
    print("Processing link:", link)

    if not link or (isinstance(link, str) and link.strip().upper() == "N/A"):
        return None

    if not isinstance(link, str):
        link = str(link)

    link = link.strip().rstrip("/")
    parts = link.split("/")

    if parts[-1].lower() == "index.html":
        return parts[-2].lower() if len(parts) >= 2 else None
    else:
        return parts[-1].lower()


def update_csv_with_dblp_ids(input_csv_path, output_csv_path):
    """
    Reads a CSV file, extracts DBLP conference IDs from the 'DBLP Source' column,
    adds a new 'DBLP ID' column, and saves the updated DataFrame to a new CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the updated CSV file.
    """
    try:
        # Read CSV without converting "N/A" to NaN
        df = pd.read_csv(input_csv_path, sep=",", header=0, keep_default_na=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Extract DBLP IDs row-wise
    df["DBLP ID"] = df["DBLP Source"].apply(extract_dblp_conference_id)

    # Save the updated DataFrame
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to '{output_csv_path}'.")


def fetch_dblp_source_link(conference_id):
    """
    Retrieves the DBLP Source link for a given conference ID from the CORE portal.

    Args:
        conference_id (str): The conference ID (first column in the CSV).

    Returns:
        str or None: The DBLP Source link if found, otherwise None.
    """
    url = f"https://portal.core.edu.au/conf-ranks/{conference_id}/"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all <div> elements with class "row oddrow"
        divs = soup.find_all("div", class_="row oddrow")

        for div in divs:
            text = div.get_text(separator=" ", strip=True)
            if "DBLP Source:" in text:
                # Remove the label to keep only the link
                source_link = text.replace("DBLP Source:", "").strip()
                return source_link

        return None  # Return None if no matching block is found

    except Exception as e:
        print(f"Error retrieving page {url}: {e}")
        return None
    

def update_csv_with_dblp_sources(input_csv_path, output_csv_path):
    """
    Reads the CORE CSV file, fetches the DBLP Source link for each conference ID,
    adds a new column 'DBLP Source', and saves the updated DataFrame to a new CSV.

    Args:
        input_csv_path (str): Path to the original CORE CSV file.
        output_csv_path (str): Path where the updated CSV will be saved.
    """
    try:
        # Read the CSV file with specified column names
        df = pd.read_csv(
            input_csv_path, 
            sep=",", 
            header=None, 
            names=["ID", "Conference Name", "Acronym", "Source Year", "Rank", "Included", "Code1", "Code2", "Code3"]
        )
    except Exception as e:
        print(f"Error reading CORE CSV file: {e}")
        return

    dblp_sources = []
    for idx, row in df.iterrows():
        conference_id = str(row["ID"]).strip()
        print(f"Processing ID {conference_id}...")
        source_link = fetch_dblp_source_link(conference_id)
        dblp_sources.append(source_link)
        time.sleep(1)  # Sleep to avoid overloading the server

    df["DBLP Source"] = dblp_sources
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to '{output_csv_path}'.")


def update_gunning_fog_all_documents(collection):
    try:
        count = 0
        while True:
            doc = collection.find_one({
                "summary": {"$exists": True},
                "summary_gunning_fog_score": {"$exists": False}
            }, {"_id": 1, "summary": 1})

            if not doc:
                print("All documents processed.")
                break

            summary = doc.get("summary", "")
            score = textstat.gunning_fog(summary)

            if score < 6:
                category = "Very Easy"
            elif score < 9:
                category = "Easy"
            elif score < 12:
                category = "Moderate"
            elif score < 15:
                category = "Hard"
            else:
                category = "Very Hard"

            collection.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "summary_gunning_fog_score": score,
                        "summary_gunning_fog_category": category
                    },
                    "$unset": {
                        "flesch_kincaid_score": "",
                        "flesch_kincaid_category": ""
                    }
                }
            )

            count += 1
            if count % 50 == 0:
                print(f"{count} documents updated...")

        print(f"Finished updating {count} documents.")

    except Exception as e:
        print(f"Error: {e}")


def process_documents_add_hindex(collection):
    import sys, codecs
    # Ensure UTF-8 output
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

    query = {"authors": {"$exists": True}, "hindex": {"$exists": False}}
    total_docs = collection.count_documents(query)
    print(f"Total documents to process: {total_docs}")

    doc_counter = 0

    try:
        while True:
            doc = collection.find_one(query, {"_id": 1, "authors": 1})
            if not doc:
                print("All documents processed.")
                break

            doc_counter += 1
            doc_id = doc["_id"]
            authors = doc.get("authors", [])

            print(f"\nProcessing document {doc_counter} - ID: {doc_id}")

            hindexes = []

            for j, author in enumerate(authors, start=1):
                print(f"  Author {j}/{len(authors)}: {author}")
                try:
                    search_query = scholarly.search_author(author)
                    author_data = next(search_query)
                    author_info = scholarly.fill(author_data)
                    hindex = author_info.get('hindex', 0) or 0
                except Exception as e:
                    print(f"    Error retrieving h-index for {author}: {e}")
                    hindex = 0
                print(f"    H-index: {hindex}")
                hindexes.append(hindex)

            collection.update_one({"_id": doc_id}, {"$set": {"hindex": hindexes}})
            print(f"Document {doc_counter} updated.")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def add_hindex_average(collection):
    query = {"hindex": {"$exists": True}, "hindex_average": {"$exists": False}}
    docs = collection.find(query)

    count = 0

    for doc in docs:
        hindex_list = doc.get("hindex", [])

        if isinstance(hindex_list, list) and len(hindex_list) > 0:
            hindex_avg = sum(hindex_list) / len(hindex_list)
        else:
            hindex_avg = 0.0

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"hindex_average": hindex_avg}}
        )
        count += 1
        print(f" Document {count} updated with hindex_average = {hindex_avg:.2f}")

    print(f"\nFinished: {count} documents updated.")


def add_version_and_image(collection):
    while True:
        doc = collection.find_one({
            "version": {"$exists": False},
            "image": {"$exists": False}
        })

        if doc is None:
            print(" All documents have been processed.")
            break

        update_fields = {}

        # Extract version from 'id' field (e.g., arXiv id like "2101.12345v2")
        arxiv_id = doc.get("id", "")
        match = re.search(r'v(\d+)$', arxiv_id)
        if match:
            version = int(match.group(1))
            update_fields["version"] = version
        else:
            print(f" No version found for document {doc.get('_id')}")

        # Count images in PDF using fitz
        pdf_path = doc.get("PDFPath", "")
        try:
            doc_pdf = fitz.open(pdf_path)
            image_count = sum(len(page.get_images(full=True)) for page in doc_pdf)
            update_fields["image"] = image_count
            doc_pdf.close()
        except Exception as e:
            print(f" Error with PDF {pdf_path}: {e}")
            update_fields["image"] = 0

        # Update document in MongoDB
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_fields}
        )
        print(f" Document {doc['_id']} updated with {update_fields}")


def count_references_in_listbibl(pdf_path, grobid_url="http://localhost:8070/api/processFulltextDocument"):
    """
    Sends a PDF to the GROBID service to extract bibliographic references,
    and counts the number of <biblStruct> entries inside the <listBibl> section of the TEI XML response.

    Args:
        pdf_path (str): Path to the PDF file.
        grobid_url (str): URL of the GROBID service API endpoint.

    Returns:
        int: Number of bibliographic references found.
    """
    with open(pdf_path, 'rb') as f:
        files = {'input': (pdf_path, f, 'application/pdf')}
        response = requests.post(grobid_url, files=files)

    if response.status_code != 200:
        raise Exception(f"GROBID error: {response.status_code} - {response.text}")

    tei_xml = response.text
    root = etree.fromstring(tei_xml.encode('utf-8'))

    # Find all <biblStruct> inside <listBibl> in TEI namespace
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    list_bibls = root.xpath('//tei:TEI//tei:text//tei:listBibl//tei:biblStruct', namespaces=namespaces)

    return len(list_bibls)


def add_references_to_all_documents(collection):
    """
    For each document in the given MongoDB collection missing the 'reference' field,
    counts the number of bibliographic references in its PDF using GROBID,
    then updates the document with this count.

    Args:
        collection (pymongo.collection.Collection): MongoDB collection to process.
    """
    documents_to_process = list(collection.find({"reference": {"$exists": False}}))

    for doc in documents_to_process:
        pdf_path = doc.get("PDFPath", "")
        if not pdf_path:
            print(f"Document {doc['_id']} has no PDF path.")
            continue

        try:
            nb_refs = count_references_in_listbibl(pdf_path)
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"reference": nb_refs}}
            )
            print(f"Document {doc['_id']} updated with {nb_refs} references.")
        except Exception as e:
            print(f"Error processing document {doc['_id']}: {e}")

        time.sleep(1)  # To avoid server overload

PROMPT = {
    "review_with_tag_score": ["You are a professor in computer science, machine learning and artificial intelligence. Write an academic-style review of the following scientific article using the following tags to structure your response: [MOTIVATION], [SUBSTANCE], [ORIGINALITY], [SOUNDNESS], [CLARITY], [STRENGTHS], [WEAKNESSES]. Each tag should introduce a short paragraph focused on that specific aspect. The review must not exceed 700 tokens, but it is not necessary to reach this limit if the content does not require it. Maintain a formal and objective tone suitable for an academic setting.", "Based on the attached scientific article and the review text, assign each scores from 1 to 5 (higher means better). Output format (no deviations, no extra words, no punctuation other than shown below, no field renaming, no explanations): SOUNDNESS_CORRECTNESS: x, ORIGINALITY: x, CLARITY: x, RELEVANCE: x, METHODOLOGY: x"],
    "review_without_tag_score": ["You are a professor in computer science, machine learning and artificial intelligence. Write an academic-style review of the following scientific article. The review should include a concise summary of the paper’s objectives, methods, results, and conclusions. Critically evaluate the strengths and weaknesses of the study, discuss its originality and contribution to the field, and suggest possible improvements or future research directions. Maintain a formal and objective tone suitable for an academic setting. The review should not exceed 700 tokens, and it is not necessary to reach this limit if the content does not require it.", "Based on the attached scientific article and the review text, assign each scores from 1 to 5 (higher means better). Output format (no deviations, no extra words, no punctuation other than shown below, no field renaming, no explanations): SOUNDNESS_CORRECTNESS: x, ORIGINALITY: x, CLARITY: x, RELEVANCE: x, METHODOLOGY: x"],
    "score_with_review": ["You are a professor in computer science, machine learning and artificial intelligence. Write an academic-style review of the following scientific article. The review should include a concise summary of the paper’s objectives, methods, results, and conclusions. Critically evaluate the strengths and weaknesses of the study, discuss its originality and contribution to the field, and suggest possible improvements or future research directions. Maintain a formal and objective tone suitable for an academic setting. The review should not exceed 700 tokens, and it is not necessary to reach this limit if the content does not require it.", "Based on the attached scientific article and the review text assign an overall evaluation score from 1 to 10 (10 being the highest). Only output the score, nothing else."],
    "score_without_review": ["You are a professor in computer science, machine learning and artificial intelligence. Based on the attached scientific article assign an overall evaluation score from 1 to 10 (10 being the highest). Only output the score, nothing else."],
    "prediction_with_review": ["You are a professor in computer science, machine learning and artificial intelligence. Write an academic-style review of the following scientific article. The review should include a concise summary of the paper’s objectives, methods, results, and conclusions. Critically evaluate the strengths and weaknesses of the study, discuss its originality and contribution to the field, and suggest possible improvements or future research directions. Maintain a formal and objective tone suitable for an academic setting. The review should not exceed 700 tokens, and it is not necessary to reach this limit if the content does not require it.","Based on the attached scientific article and the review text, predict the likely editorial decision for an academic conference. Choose only one of the following options and output only that decision: Accept,Reject, Minor Revision, Major Revision"],
    "prediction_without_review": ["Based on the attached scientific article predict the likely editorial decision for an academic conference. Choose only one of the following options and output only that decision: Accept, Reject, Minor Revision, Major Revision"],
    "percentage_prediction_with_review": ["You are a professor in computer science, machine learning and artificial intelligence. Write an academic-style review of the following scientific article. The review should include a concise summary of the paper’s objectives, methods, results, and conclusions. Critically evaluate the strengths and weaknesses of the study, discuss its originality and contribution to the field, and suggest possible improvements or future research directions. Maintain a formal and objective tone suitable for an academic setting. The review should not exceed 700 tokens, and it is not necessary to reach this limit if the content does not require it.","Based on the attached scientific article and its review, estimate as precisely as possible the percentile ranking this paper would likely achieve among all submissions, assuming a realistic distribution of submission quality. A lower percentile indicates a better paper (e.g., 5 = top 5%). Output only a single integer between 1 and 100. No explanation or additional text."],
    "percentage_prediction_without_review": ["Based on the attached scientific article, estimate as precisely as possible the percentile ranking this paper would likely achieve among all submissions, assuming a realistic distribution of submission quality. A lower percentile indicates a better paper (e.g., 5 = top 5%). Output only a single integer between 1 and 100. No explanation or additional text."]
}

def global_prompt(prompt, collection, output_path, openai_api_key):
    """
    Processes articles from a MongoDB collection using OpenAI API with specified prompts,
    and writes the results (reviews, scores, predictions) into an output file.

    Args:
        prompt (list or str): A single prompt string or a list of two prompts (e.g. [prompt_1, prompt_2])
                              for a two-step process (review + scoring).
        collection (pymongo.collection.Collection): MongoDB collection with articles.
        output_path (str): Path to the output text file where results will be saved.
        openai_api_key (str): API key for OpenAI client authentication.

    Returns:
        None
    """
    openai_client = OpenAI(api_key=openai_api_key)

    # Retrieve 3 random PDF paths for articles ranked "A*"
    a_star_paths = [
        doc["PDFPath"] for doc in collection.aggregate([
            {"$match": {"core_publication_rank": "A*"}},
            {"$sample": {"size": 3}},
            {"$project": {"_id": 0, "PDFPath": 1}}
        ])
    ]

    # Retrieve 3 random PDF paths for articles ranked "C"
    c_paths = [
        doc["PDFPath"] for doc in collection.aggregate([
            {"$match": {"core_publication_rank": "C"}},
            {"$sample": {"size": 3}},
            {"$project": {"_id": 0, "PDFPath": 1}}
        ])
    ]

    # Combine paths from both ranks
    articles_paths = a_star_paths + c_paths

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for path in articles_paths:
            # Upload the PDF file to OpenAI for processing
            uploaded_file = openai_client.files.create(
                file=open(path, "rb"),
                purpose="user_data"
            )

            # Retrieve the article document based on PDFPath
            article = collection.find_one({"PDFPath": path}, {"_id": 1, "core_publication_rank": 1})

            article_id = article["_id"]
            publication_rank = article.get("core_publication_rank", "Unknown")

            # Write article metadata to output file
            output_file.write(f"ID: {article_id}\n")
            output_file.write(f"Core Publication Rank: {publication_rank}\n\n")

            # Handle two-step prompt (e.g. review + scoring)
            if isinstance(prompt, list) and len(prompt) == 2:
                # 1. Generate the review text
                review_response = openai_client.responses.create(
                    model="gpt-4o-mini",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file.id},
                            {"type": "input_text", "text": prompt[0]}
                        ]
                    }]
                )
                review_text = review_response.output[0].content[0].text

                # 2. Generate the score or secondary output based on the review
                score_response = openai_client.responses.create(
                    model="gpt-4o-mini",
                    instructions=prompt[1],
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file.id},
                            {"type": "input_text", "text": review_text}
                        ]
                    }]
                )
                score_output = score_response.output[0].content[0].text

                # Write the review and score output to the file
                output_file.write("Review:\n")
                output_file.write(review_text.strip() + "\n")
                output_file.write("Score Output:\n")
                output_file.write(score_output.strip() + "\n\n")

            else:
                # Single prompt case
                simple_response = openai_client.responses.create(
                    model="gpt-4o-mini",
                    instructions="You are a professor in computer science, machine learning and artificial intelligence.",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file.id},
                            {"type": "input_text", "text": prompt if isinstance(prompt, str) else prompt[0]}
                        ]
                    }]
                )
                output_text = simple_response.output[0].content[0].text

                output_file.write("Output:\n")
                output_file.write(output_text.strip() + "\n\n")

            # Sleep 1 second between requests to avoid rate limits
            time.sleep(1)
