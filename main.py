import os
import logging
from ingestion import DataIngestor
from indexing import DocumentIndexer
from config import settings

# Setup logging for the main entry point
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_index():
    """
    Orchestrates the full pipeline:
    1. Scans 'data/' folder and processes files (Multimodal + Text).
    2. Converts those files into vector embeddings.
    3. Saves them into the ChromaDB local folder.
    """
    logger.info("Starting Full RAG Ingestion & Indexing Pipeline...")

    # 1. Initialize Ingestor (Handles PDF, Images, Video, Audio)
    ingestor = DataIngestor()
    chunks = ingestor.run_pipeline()
    
    if not chunks:
        logger.warning("No document chunks were generated. Check your 'data/' folder.")
        return

    # 2. Initialize Indexer (Handles ChromaDB & Embedding Model)
    indexer = DocumentIndexer()
    
    # 3. Save to Disk
    # This will create the 'chroma_db' folder if it doesn't exist
    indexer.save_to_disk(chunks)
    
    logger.info(f"Successfully indexed {len(chunks)} chunks.")
    logger.info("You can now run the app using: streamlit run app.py")

if __name__ == "__main__":
    # Ensure the data directory exists before running
    if not os.listdir(settings.paths.DATA_DIR):
        print(f"The data folder '{settings.paths.DATA_DIR}' is empty.")
        print("Please drop some files (PDFs, Images, MP4s) there first!")
    else:
        rebuild_index()