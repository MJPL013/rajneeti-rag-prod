import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent))

from src.core.config import settings
from src.core.logger import logger
from src.ingestion.loader import DataLoader
from src.ingestion.vector_store import VectorDBManager

def main():
    logger.info("Starting Ingestion Pipeline...")
    
    # Initialize components
    loader = DataLoader(settings.RAW_DATA_DIR)
    vector_db = VectorDBManager()
    
    batch_size = 100
    batch = []
    
    count = 0
    for statement in loader.load_data():
        batch.append(statement)
        if len(batch) >= batch_size:
            vector_db.upsert_statements(batch)
            count += len(batch)
            batch = []
            
    # Upsert remaining
    if batch:
        vector_db.upsert_statements(batch)
        count += len(batch)
        
    logger.info(f"Ingestion complete. Total statements processed: {count}")

if __name__ == "__main__":
    main()
