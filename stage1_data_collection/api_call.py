import os
import json
import logging
import time
import google.generativeai as genai
from tqdm import tqdm
import sys
import concurrent.futures

# --- CONFIGURATION ---
# IMPORTANT: Set your Gemini API Key as an environment variable named 'GEMINI_API_KEY'
# or paste it directly here.
API_KEY = os.getenv("GEMINI_API_KEY")

# Define the input and output directories
# INPUT_DIR  : Folder containing raw article JSONs (one file per politician)
# OUTPUT_DIR : Folder where Gemini-processed statement JSONs will be saved (one file per article_id)
INPUT_DIR = "stage1_data_collection/raw_articles"
OUTPUT_DIR = "stage1_data_collection/gemini_outputs"
LOG_DIR = "stage1_data_collection/gemini_outputs"
PROMPT_FILE_PATH = "stage1_data_collection"

# --- API & PARALLELISM SETTINGS ---
# Number of parallel threads to run. A safe number is 5-10.
MAX_WORKERS = 10
# Delay in seconds before each API call to avoid hitting rate limits (e.g., 60 requests/minute).
DELAY_BETWEEN_REQUESTS = 1

# List of input files to process. You can add or remove files from this list.
# INPUT_FILES = [
#     "Final_Arvind_Kejriwal_V1.json",
#     "Final_Mamata_Banerjee_V1.json",
#     "Final_MK_Stalin_V1.json",
#     "Final_Pinarayi_Vijayan_V1.json",
#     "Final_Yogi_Adityanath_V1.json"
# ]

INPUT_FILES = [
    "Final_MK_Stalin_V1.json"
]


# --- LOAD PROMPT FROM EXTERNAL FILE ---
try:
    sys.path.append(os.path.abspath(PROMPT_FILE_PATH))
    from prompt import TRAINING_SET_PROMPT
except ImportError:
    print(f"Error: Could not import TRAINING_SET_PROMPT from prompt.py.")
    print(f"Please ensure 'prompt.py' exists in the '{PROMPT_FILE_PATH}' directory.")
    sys.exit(1)


def setup_logging():
    """Configures the logging to save to a file and print to console."""
    log_file = os.path.join(LOG_DIR, "processing_log.txt")
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 'a' for append mode
            logging.StreamHandler(sys.stdout)
        ]
    )

def process_article(article, model):
    """
    Processes a single article: prepares prompt, calls Gemini API, and returns the result.
    Returns the article_id and the processed data, or None if failed.
    """
    article_id = article.get("id")
    try:
        metadata = article.get("metadata", {})
        publish_date = metadata.get("publish_date") if metadata.get("publish_date") else metadata.get("year", "N/A")

        final_prompt = TRAINING_SET_PROMPT.format(
            article_id=str(article.get('id', 'N/A')),
            politician=str(metadata.get('politician', 'N/A')),
            source=str(metadata.get('site_name', 'N/A')),
            publish_date=str(publish_date),
            url=str(metadata.get('url', 'N/A')),
            title=str(metadata.get('title', 'N/A')),
            text=str(article.get('content', ''))
        )
        
        # Add a delay to respect rate limits
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Call the Gemini API
        response = model.generate_content(final_prompt)
        
        # Clean the response to extract only the JSON part
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        
        gemini_output = json.loads(response_text)
        return article_id, gemini_output

    except Exception as e:
        logging.error(f"FAILED to process article ID: {article_id}. Error: {e}")
        return article_id, None


def main():
    """Main function to orchestrate the processing of article files using parallel execution."""
    setup_logging()
    logging.info("==========================================================")
    logging.info(f"Starting new article processing session with {MAX_WORKERS} parallel workers.")
    logging.info("==========================================================")

    if not API_KEY:
        logging.error("GEMINI_API_KEY environment variable not set. Exiting.")
        return

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_articles_found = 0
    total_success = 0
    total_failed = 0
    total_skipped = 0

    for filename in INPUT_FILES:
        input_filepath = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(input_filepath):
            logging.warning(f"Input file not found: {input_filepath}. Skipping.")
            continue

        logging.info(f"\n--- Processing input file: {filename} ---")
        
        try:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Could not decode JSON from {filename}. Error: {e}. Skipping file.")
            continue
        
        articles_to_process = []
        file_skipped = 0
        
        # First, determine which articles need processing vs. skipping
        for article in articles:
            article_id = article.get("id")
            if not article_id:
                logging.warning("Article missing 'id', it will be skipped.")
                continue
            
            output_filepath = os.path.join(OUTPUT_DIR, f"{article_id}.json")
            if os.path.exists(output_filepath):
                file_skipped += 1
            else:
                articles_to_process.append(article)
        
        total_articles_found += len(articles)
        total_skipped += file_skipped
        logging.info(f"Found {len(articles)} total articles. {len(articles_to_process)} need processing, {file_skipped} already exist.")

        file_success = 0
        file_failed = 0

        # Process the articles in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_article = {executor.submit(process_article, article, model): article for article in articles_to_process}
            
            progress_bar = tqdm(concurrent.futures.as_completed(future_to_article), total=len(articles_to_process), desc=f"Analyzing {filename}", unit="article")

            for future in progress_bar:
                article_id, result = future.result()
                
                if result:
                    output_filepath = os.path.join(OUTPUT_DIR, f"{article_id}.json")
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    file_success += 1
                else:
                    file_failed += 1
        
        logging.info(f"--- Finished processing {filename} ---")
        logging.info(f"Summary for {filename}: Success={file_success}, Failed={file_failed}, Skipped={file_skipped}")
        
        total_success += file_success
        total_failed += file_failed

    logging.info("\n==========================================================")
    logging.info("          MASTER PROCESSING REPORT          ")
    logging.info("==========================================================")
    logging.info(f"Total articles across all files: {total_articles_found}")
    logging.info(f"Successfully processed (new):    {total_success}")
    logging.info(f"Skipped (already existed):       {total_skipped}")
    logging.info(f"Failed to process:               {total_failed}")
    logging.info("==========================================================")


if __name__ == "__main__":
    main()
