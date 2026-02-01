#indexing.py
import os
import logging
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import settings

# Setup logging consistent with the project
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self):
        # 1. Setup the local embedding model
        logger.info(f"Initializing embedding model: {settings.models.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.models.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} 
        )

    def save_to_disk(self, documents: List[Document]):
        """
        Takes a list of document chunks, embeds them, and saves to ChromaDB.
        """
        if not documents:
            logger.warning("No documents provided.")
            return
        
        try:
            # Check if the directory exists and has files (indicating an existing DB)
            db_exists = os.path.exists(settings.paths.VECTOR_DB_DIR) and \
                        len(os.listdir(settings.paths.VECTOR_DB_DIR)) > 0

            if db_exists:
                logger.info(f"Appending {len(documents)} chunks to existing index...")
                vectorstore = self.load_existing_index()
                vectorstore.add_documents(documents) # This is the "Append" command
            else:
                logger.info(f"Creating NEW index with {len(documents)} chunks...")
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=settings.paths.VECTOR_DB_DIR,
                    collection_name=settings.vector_db.COLLECTION_NAME,
                    collection_metadata={"hnsw:space": settings.vector_db.DISTANCE_FUNCTION}
                )
            
            logger.info("✅ Database updated successfully.")
            return vectorstore
                
        except Exception as e:
            logger.error(f"Failed to update vector index: {e}")
            raise
        
    def load_existing_index(self):
        """
        Loads the database from the disk without creating new embeddings.
        """
        logger.info("Loading existing vector index from disk...")
        return Chroma(
            persist_directory=settings.paths.VECTOR_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=settings.vector_db.COLLECTION_NAME
        )

