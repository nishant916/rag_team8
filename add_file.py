# add_file.py
# Script to add a single new file to the existing ChromaDB index
from ingestion import DataIngestor
from indexing import DocumentIndexer

# path to your one new file
new_file = "data/github_project_page.html"

ingestor = DataIngestor()
new_chunks = ingestor.process_single_file(new_file)

if new_chunks:
    indexer = DocumentIndexer()
    indexer.save_to_disk(new_chunks)