# config.py
# Defines all configuration settings for the RAG system using Pydantic.

import os
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class PathSettings(BaseModel):
    """Configuration for all file and directory paths."""
    DATA_DIR: str = "data"
    VECTOR_DB_DIR: str = "chroma_db"
    LOG_FILE: str = "rag_system.log"

class GoogleSettings(BaseModel):
    """Configuration specifically for Google/Gemini API."""
    GOOGLE_API_KEY: str = Field(..., validation_alias=AliasChoices("GOOGLE__GOOGLE_API_KEY", "GOOGLE_API_KEY")) # ellipsis (...) makes this field REQUIRED in .env

class ModelSettings(BaseModel):
    """Configuration for LLM and Embedding models."""
    
    # --- LLM Settings ---
    LLM_MODEL: str = "gemini-3-flash-preview"
    LLM_TEMPERATURE: float = 0.5    # Controls randomness in output (0.0 = deterministic, 1.0 = very random), Gemini default = 1.0
    LLM_MAX_OUTPUT_TOKENS: int = 4096  # Max tokens in output response, Gemini default = 65536
    
    # --- Generic system prompt specifying the purpose/role, constrainsts, example and fall-back ---
    """
    LLM_SYSTEM_PROMPT: str = (
        "You are an expert AI assistant. Answer the user's questions truthfully and only using the provided context. "
        "If you use any information from the context, you MUST cite the source document(s) "
        "by listing their 'source' metadata field at the end of your response, e.g., (Source: paper_title.pdf)."
        "If the answer is not available in the context, state 'I cannot answer this based on the provided documents.'"
    )"""

    LLM_SYSTEM_PROMPT: str = (
    "You are an expert AI assistant. Your goal is to provide comprehensive answers by "
    "integrating the provided context with your own internal knowledge. "
    "\n\nGUIDELINES:"
    "\n1. PRIORITIZE the provided context for specific facts, technical details, and data."
    "\n2. SUPPLEMENT with your own knowledge to provide broader explanations, definitions, or "
    "to fill in logical gaps not covered by the documents."
    "\n3. CITATION: If you use information from the provided context, you MUST cite the source "
    "at the end of your response (e.g., Source: file_name.pdf)."
    "\n4. DISTINCTION: If you provide information NOT found in the context, clearly state "
    "'[Additional Context]' or similar to distinguish it from the source data."
    )

    BASE_LLM_SYSTEM_PROMPT: str = "You are a helpful, knowledgeable, and reliable general assistant. Answer the user's question using your vast, general knowledge."

    # --- Embedding Settings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"    # dimensions: 384


class DocumentSettings(BaseModel):
    """Configuration for document loading and chunking."""
    
    # Unstructured Loader Settings
    UNSTRUCTURED_STRATEGY: str = "hi_res" # 'hi_res' for detailed analysis
    UNSTRUCTURED_OCR_ENABLED: bool = True # Enable OCR for image-based PDFs

    # Chunking Control (LangChain RecursiveCharacterTextSplitter)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

class VectorDBSettings(BaseModel):
    """Configuration for the Vector Database"""
    
    COLLECTION_NAME: str = "rag_document_collection"
    DISTANCE_FUNCTION: str = "cosine" # or "l2" (Euclidean), "ip" (Inner Product)
    K_RETRIEVAL: int = 5  # Number of similar documents to retrieve during querying

class Settings(BaseSettings):
    """Main configuration object aggregating all settings. Pydantic will look for environment variables matching these names (case-insensitive)"""
    
    paths: PathSettings = PathSettings()    
    models: ModelSettings = ModelSettings()
    documents: DocumentSettings = DocumentSettings()
    vector_db: VectorDBSettings = VectorDBSettings()

    google: GoogleSettings

    # Pydantic v2 specific configuration
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore',
        env_nested_delimiter='__' # Allows setting models__LLM_TEMPERATURE=0.5 in .env
    )

# Instantiate the settings object
settings = Settings()

# Ensure necessary directories exist on startup
os.makedirs(settings.paths.DATA_DIR, exist_ok=True)
os.makedirs(settings.paths.VECTOR_DB_DIR, exist_ok=True)