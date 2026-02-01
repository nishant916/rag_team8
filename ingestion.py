# ingestion.py
# Ingests various document types (text, PDFs, MS-Office documents, images, audio, video etc.) and processes them into text chunks

import os
import logging
import mimetypes                           # For file type/subtype detection i.e., file type (e.g. video, audio, image) / format (.mp4, .mp3, .jpg)
import base64                              # For encoding binary values (images, videos) to base64 (A-Z, a-z, 0-9, +, /) strings
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage                # Represents user input and interactions with the model (could be multimodal)
from bs4 import BeautifulSoup               # For HTML parsing and content extraction

# Import config
from config import settings
from logger_config import setup_app_logging

# --- Logging Configuration ---
setup_app_logging(mode='w')
logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self):
        # 1. Setup Chunking logic for text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.documents.CHUNK_SIZE,
            chunk_overlap=settings.documents.CHUNK_OVERLAP
        )
        
        # 2. Initialize Gemini-3-Flash for Multimodal processing
        self.multimodal_llm = ChatGoogleGenerativeAI(
            model=settings.models.LLM_MODEL,
            google_api_key=settings.google.GOOGLE_API_KEY,
            temperature=settings.models.LLM_TEMPERATURE
        )

    def process_media_with_gemini(self, file_path: str, mime_type: str) -> str:
        """
        Sends Image, Video, or Audio to Gemini to generate a text representation.
        Ensures the output is a plain string for LangChain Document compatibility.
        """
        logger.info(f"Processing multimodal file ({mime_type}): {file_path}")
        
        try:
            # 1. Read and encode the file bytes
            with open(file_path, "rb") as f:           # Read in binary mode
                file_bytes = f.read()
                # Encode to base64 string
                encoded_file = base64.b64encode(file_bytes).decode("utf-8")

            # 2. Tailor the prompt based on file type
            if mime_type.startswith('image/'):
                prompt = "Describe this image in detail. Extract any visible text and objects for a search index."
            elif mime_type.startswith('video/'):
                prompt = "Provide a comprehensive transcript and visual summary of this video. Include timestamps and speaker labels."
            elif mime_type.startswith('audio/'):
                prompt = "Transcribe this audio file accurately and summarize the main topics discussed."
            else:
                prompt = "Describe the contents of this file in detail for a retrieval system."

            # 3. Create the multimodal message
            # We use the 'data' key for local files instead of 'file_uri'
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "media", 
                        "mime_type": mime_type, 
                        "data": encoded_file 
                    }
                ]
            )
            
            # 4. Invoke the model
            response = self.multimodal_llm.invoke([message])
            
            # 5. CRITICAL: Extract only the string content
            # This ensures the 'page_content' in your Document is a valid string
            if hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM processing failed for {file_path}: {e}")
            # Return a descriptive string so the indexer doesn't crash
            return f"Error: The system could not process the file {os.path.basename(file_path)}. Exception: {str(e)}"

    def load_single_file(self, file_path: str) -> List[Document]:
        """
        Determines the file type and routes it to the correct processor.
        """
        mime_type, _ = mimetypes.guess_type(file_path)  # returns a tuple: ('type/subtype', encoding)
        
        # A. Handle Multimodal (Images, Video, Audio)
        multimodal_types = ['image/', 'video/', 'audio/']
        if mime_type and any(mime_type.startswith(t) for t in multimodal_types):
            # We call our Gemini processor
            description = self.process_media_with_gemini(file_path, mime_type)

            logger.info(f"Successfully generated text description for {os.path.basename(file_path)}")

            return [Document(
                page_content=description, 
                metadata={"source": file_path, "type": "media_description"}
            )]
        
        # B. Handle HTML Documents
        if mime_type == "text/html":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')

                # 1. Process Tables into a readable grid format
                for table in soup.find_all('table'):
                    rows = []
                    for tr in table.find_all('tr'):
                        # Extract text from both header (th) and data (td) cells
                        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                        if cells:
                            rows.append(" | ".join(cells))
                    
                    if rows:
                        table_text = "\n\n[TABLE START]\n" + "\n".join(rows) + "\n[TABLE END]\n\n"
                        table.replace_with(table_text)

                # 2. Process Images (Stick to Alt-Text as requested)
                for img in soup.find_all('img'):
                    alt_text = img.get('alt', 'No description available')
                    img_placeholder = f" [IMAGE CONTENT: {alt_text}] "
                    img.replace_with(img_placeholder)

                # 3. Extract Text with Double Newlines
                # This ensures headers and paragraphs stay separated for the splitter
                final_content = soup.get_text(separator="\n\n", strip=True)
                
                logger.info(f"Successfully cleaned HTML via BeautifulSoup: {os.path.basename(file_path)}")
                return [Document(
                    page_content=final_content, 
                    metadata={"source": file_path, "type": "text/html"}
                )]

            except Exception as e:
                logger.error(f"BS4 processing failed for {file_path}: {e}")
                return []
        
        # C. Handle Other Text Documents (PDF, DOCX, TXT etc.)
        try:
            # Unstructured for documents
            loader = UnstructuredLoader(
                file_path,
                strategy=settings.documents.UNSTRUCTURED_STRATEGY,
                chunking_strategy="by_title"   # Better structure preservation
            )
            return loader.load()
        except Exception as e:
            logger.error(f"Unstructured failed for {file_path}: {e}")
            return []

    def process_single_file(self, file_path: str) -> List[Document]:
        """
        Loads and chunks a specific file, returning cleaned chunks. Useful for adding single files to an existing index.
        """
        logger.info(f"Ingesting single file: {file_path}")
        raw_docs = self.load_single_file(file_path)
        
        if not raw_docs:
            return []
            
        chunks = self.text_splitter.split_documents(raw_docs)
        cleaned_chunks = filter_complex_metadata(chunks)
        
        logger.info(f"Generated {len(cleaned_chunks)} chunks for {os.path.basename(file_path)}")
        return cleaned_chunks

    def run_pipeline(self) -> List[Document]:
        """
        Main execution: Scans data folder, loads files, and splits text into chunks.
        """
        logger.info(f"Starting ingestion from: {settings.paths.DATA_DIR}")
        all_processed_chunks = []
        
        # Loop through the data directory
        for filename in os.listdir(settings.paths.DATA_DIR):
            file_path = os.path.join(settings.paths.DATA_DIR, filename)
            
            if os.path.isfile(file_path):
                # 1. Load the file (result is a list of Document objects)
                raw_docs = self.load_single_file(file_path)
                
                # 2. Split documents into chunks for the Vector DB
                if raw_docs:
                    chunks = self.text_splitter.split_documents(raw_docs)
                    all_processed_chunks.extend(chunks)
                    logger.info(f"Successfully processed {filename} into {len(chunks)} chunks.")
                else:
                    logger.warning(f"Skipping {filename}: No content extracted.")

        if all_processed_chunks:
            logger.info("Cleaning metadata for ChromaDB compatibility...")
            cleaned_chunks = filter_complex_metadata(all_processed_chunks)
            logger.info(f"Total Chunks Prepared and Cleaned: {len(cleaned_chunks)}")
            return cleaned_chunks
        return []
