# ArXiv to DBLP to CORE Ranking System

A comprehensive system to match academic papers from ArXiv to their corresponding DBLP entries and CORE rankings, with PDF processing and classification capabilities.

## Overview

This project builds a pipeline for:

1. Matching ArXiv papers with their DBLP entries
2. Associating DBLP conferences/journals with CORE rankings
3. Processing and analyzing PDFs from ArXiv
4. Training a machine learning model to predict conference/journal rankings

## System Architecture

The system consists of several key modules:

### 1. Matching Pipeline

- `matching/match_arxiv_dblp.py`: Matches ArXiv papers to DBLP entries using title/author searches
- `matching/match_dblp_core.py`: Maps DBLP conference/journal identifiers to CORE rankings
- `matching/update_arxiv_dblp_matching.py`: Processes batches of articles in MongoDB for ArXiv-DBLP matching
- `matching/update_dblp_core_matching.py`: Processes batches of articles in MongoDB for DBLP-CORE matching

### 2. PDF Processing

- `pdf_processing/parser_pdf.py`: Extracts metadata from PDFs (pages, characters, lines, images)
- `pdf_processing/pdf_windows_arxiv.py`: Automates downloading PDFs from ArXiv

### 3. Machine Learning

- `training_model/publication_predictor.py`: Trains a RandomForest classifier to predict CORE rankings

### 4. Database Enrichment

- `database/database_enrichment.py`: Comprehensive module for fetching, processing, and enriching academic paper data, including:
  - MongoDB Atlas connectivity with secure authentication
  - ArXiv API integration to fetch computer science (cs.AI) articles
  - PDF downloading and metadata extraction (pages, characters, images)
  - Text complexity analysis using Gunning Fog readability scores
  - Author h-index retrieval and aggregation
  - Citation reference counting via GROBID integration
  - OpenAI API integration for automated paper reviews and scoring
  - CORE ranking database management and DBLP conference ID mapping

## Dependencies

- **Python Libraries**: 
  - Data Processing: `pandas`, `pymongo`
  - PDF Processing: `PyMuPDF` (fitz), `pyautogui`, `PyPDF2`
  - Web Requests: `requests`, `SPARQLWrapper`, `lxml`, `BeautifulSoup`
  - Machine Learning: `scikit-learn`, `matplotlib`, `seaborn`
  - Text Analysis: `textstat`, `scholarly`
  - Text Matching: `thefuzz`
  - API Clients: `openai`
  - UI: `tqdm`

- **Data Files**:
  - `data/CORE2023.csv`: CORE conference/journal rankings

- **Database**:
  - MongoDB for storing paper metadata and matching results

- **External Services**:
  - GROBID server for reference extraction
  - OpenAI API for paper review generation

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure MongoDB is installed and running

## Usage

### 1. ArXiv to DBLP Matching

Process a batch of ArXiv papers to find their corresponding DBLP entries:

```python
from pymongo import MongoClient
from matching.update_arxiv_dblp_matching import process_arxiv_dblp_matching

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['academic_papers']
collection = db['papers']

# Process a batch of papers (default batch_size=6000)
process_arxiv_dblp_matching(collection, batch_size=1000)
```

### 2. DBLP to CORE Ranking Matching

Once ArXiv-DBLP matching is complete, match DBLP entries to CORE rankings:

```python
from pymongo import MongoClient
from matching.update_dblp_core_matching import process_dblp_core_matching

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['academic_papers']
collection = db['papers']

# Process a batch of papers (default batch_size=6000)
process_dblp_core_matching(collection, batch_size=1000)
```

### 3. PDF Analysis

Extract metadata from PDFs:

```python
from pdf_processing.parser_pdf import process_pdfs

# Process all PDFs in a directory
process_pdfs('path/to/pdf/directory', 'output_analysis.json')

# Or process a limited number of PDFs
from pdf_processing.parser_pdf import process_limited_pdfs
process_limited_pdfs('path/to/pdf/directory', 'output_analysis.json', limit=100)

# Verify analysis results
from pdf_processing.parser_pdf import verify_pdf_analysis
verify_pdf_analysis('path/to/pdf/directory', 'output_analysis.json')
```

### 4. Download ArXiv PDFs (Windows only)

```python
from pdf_processing.pdf_windows_arxiv import download_arxiv_pdf

# Download a single paper from ArXiv
download_arxiv_pdf('https://arxiv.org/abs/2105.12345', file_number=1, index=0)
```

### 5. Predict Conference/Journal Rankings

Train a model to predict CORE rankings:

```python
from pymongo import MongoClient
from training_model.publication_predictor import train_rank_classifier

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['academic_papers']
collection = db['papers']

# Train classifier with equal samples per rank
classifier = train_rank_classifier(collection)

# Or specify samples per rank
samples = {'A*': 200, 'A': 200, 'B': 150, 'C': 150, 'UnRanked': 300}
classifier = train_rank_classifier(collection, samples_per_rank=samples)
```

### 6. Database Enrichment

Various utilities for enhancing the academic paper database:

```python
from pymongo import MongoClient
from database.database_enrichment import get_articles_collection

# Connect to MongoDB Atlas
collection = get_articles_collection("username", "password")

# Fetch sample of cs.AI articles from ArXiv (years 2018-2023)
from database.database_enrichment import fetch_articles_by_years
fetch_articles_by_years()

# Add PDF file paths to documents
from database.database_enrichment import add_pdf_path
add_pdf_path(collection)

# Download PDFs from ArXiv
from database.database_enrichment import create_pdf_folder_and_download
create_pdf_folder_and_download(collection)

# Extract and update PDF metadata (pages, characters)
from database.database_enrichment import update_pdf_info_in_db
update_pdf_info_in_db(collection)

# Update readability scores for article summaries
from database.database_enrichment import update_gunning_fog_all_documents
update_gunning_fog_all_documents(collection)

# Retrieve and add author h-indexes
from database.database_enrichment import process_documents_add_hindex, add_hindex_average
process_documents_add_hindex(collection)
add_hindex_average(collection)

# Add paper version numbers and image counts
from database.database_enrichment import add_version_and_image
add_version_and_image(collection)

# Count reference citations using GROBID
from database.database_enrichment import add_references_to_all_documents
add_references_to_all_documents(collection)

# Generate AI reviews and rankings for papers
from database.database_enrichment import global_prompt, PROMPT
global_prompt(PROMPT["review_with_tag_score"], collection, "output_reviews.txt", "your_openai_api_key")
```

## MongoDB Schema

The system stores the following information in MongoDB:

### Basic Paper Info
- `_id`: MongoDB document ID
- `title`: Paper title from ArXiv
- `authors`: List of authors

### ArXiv-DBLP Matching
- `arxiv_dblp_linked`: Boolean flag indicating matching was attempted
- `matched`: Boolean indicating if a match was found
- `dblp_title`: Matched paper title in DBLP
- `dblp_conference`: Conference/journal name from DBLP
- `dblp_key`: DBLP key (e.g., "conf/models/2022")
- `title_matching_score`: Fuzzy matching score
- `matching_method`: Method used for matching ("API title search", "API author search", or "SPARQL search")

### DBLP-CORE Matching
- `dblp_core_linked`: Boolean flag indicating ranking was attempted
- `core_publication`: Matched conference/journal name in CORE
- `core_publication_score`: Fuzzy matching score for CORE publication
- `core_publication_rank`: CORE rank (A*, A, B, C, or UnRanked)
- `publication_type`: Type of publication ("conference" or "journal")

### PDF Analysis
- `abstract_size`: Size of abstract
- `character`: Number of characters in PDF
- `page`: Number of pages in PDF
- `image`: Number of images in PDF
- `PDFPath`: Local path to the PDF file

### Paper Metrics
- `number_of_authors`: Count of authors
- `version`: ArXiv version number
- `reference`: Number of references
- `summary_gunning_fog_score`: Readability score of summary
- `summary_gunning_fog_category`: Categorized readability level
- `hindex`: List of author h-indexes
- `hindex_average`: Average h-index of authors

## Matching Details

### ArXiv to DBLP Matching Methods

1. **Title Search**: Searches DBLP's API using the paper title
2. **SPARQL Query**: Uses DBLP's SPARQL endpoint for more flexible matching
3. **Author Search**: Falls back to searching by authors when title searches fail

### DBLP to CORE Matching Methods

1. **DBLP ID Match**: Extracts conference ID from DBLP key (e.g., "models" from "conf/models/2022")
2. **Fuzzy Matching**: Uses fuzzy string matching on conference names and acronyms

## Model Training

The system trains a RandomForest classifier using features extracted from papers to predict CORE rankings (A*, A, B, C, UnRanked). Feature importance analysis helps identify which aspects of papers best predict their publication venue quality.

## License



## Contributors
