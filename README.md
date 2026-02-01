# 💡 LLM-Powered RAG Chat Assistant

An interactive **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, and **Google Gemini API**. This project demonstrates the power of grounding LLM responses in local data to eliminate hallucinations and provide context-specific answers.


## 🎯 Project Overview
The core of this project is a **RAG-enhanced LLM Chat Assistant**. By indexing local documents (PDFs, Text, MS Office Documents (Word, Excel), HTML, Images, Audio & Video files etc.) into a **ChromaDB** vector store, the system can "search" your files to find the most relevant information before answering. The UI also provides a toggle to turn off-RAG and chat with LLM directly.

### Key Features
* **Intelligent Ingestion:** Uses `Unstructured`, `Beautiful Soup` and `Gemini-3-flash-preview API` to load and clean various document types.
* **Semantic Chunking:** Implements `RecursiveCharacterTextSplitter` to maintain context across document fragments.
* **Vector Search:** Leverages `ChromaDB` and `GoogleGenerativeAIEmbeddings` for high-speed similarity searches.
* **Comparison with Base LLM:** A UI toggle to switch between RAG & standard LLM pipeline to see how RAG improves accuracy of responses.
* **Persistent Storage:** Indexed documents are stored locally, so you only need to index your data once.

### Limitations
* **Media File Size Restrictions:** Uses Gemini API for converting media files to text i.e. Gemini generates image description, audio transcription & video description/ transcription. This works well for small sized files.
* **Limited Number of Requests:** Gemini API's free-tier offers limited requests in a min, hour & day. Usage may be monitored in the Google AI Studio.


## 🏗️ Technical Architecture

### Primary RAG Pipeline
1.  **Load:** Documents are pulled from the `/data` directory.
2.  **Ingest:** Document are parsed into text using specific libraries (e.g. media files are converted to text)
3.  **Split:** Large files are broken into overlapping chunks (1000 chars, 100 overlap).
4.  **Embed:** Chunks are converted into 384-dimensional vectors via `Hugging Face sentence-transformers/all-MiniLM-L6-v2`.
4.  **Index:** Vectors are indexed in `ChromaDB` with `cosine` similarity measure
5.  **Retrieve:** User queries are vectorized and matched against the most similar chunks in ChromaDB.
6.  **Augment:** The retrieved context is injected into a custom prompt.
7.  **Generate:** LLM (`Gemini`) produces a grounded response based on the provided context.


## 🛠️ Setup & Installation

### 0. Prerequisites
* **Anaconda/Conda** installed. Alternatively could setup own virtual environment.
* **Google Gemini API Key** (Get one at [Google AI Studio](https://aistudio.google.com/)).

### 1. Clone the Repository
```bash
git clone [https://github.com/nishant916/rag_team8.git](https://github.com/nishant916/rag_team8.git/)
cd rag_team8
```

### 2. Environment Setup
Since this project uses an Anaconda environment:
```bash
# Create and activate your environment (if not already done)
conda create -n rag_env python=3.10
conda activate rag_env

# Install dependencies
pip install -r requirements.txt
```
Alternatively, a virtual environment could also be created with python >=3.10 (make sure compatibility with the dependencies)

### 3. Configuration
Create a .env file in the root directory and add your API credentials:
`GOOGLE__GOOGLE_API_KEY="YOUR_API_KEY_HERE"`


## 🚀 Usage
1. **Add Data:** Place your documents (PDFs, TXT, DOCX, XLSX, CSV, HTML, AUDIO, VIDEO OR IMAGE files etc.) into the data/ folder.

2. **Ingest & Index Documents:** Run main.py
```bash
python main.py
```

3. **Launch the App:** After indexing is complete
```bash
streamlit run app.py
```
4. **Chat:** Start chatting with your chat assistant. You make check the logs for ingestion, indexing and RAG-retrieval in rag_system.log


## 📂 Project Structure
```text
rag_team8/
├── chroma_db/               # Local vector store (ignored by Git)
├── data/                    # Your raw documents (ignored by Git)
├── ingestion.py             # Class for data loading, ingestion & chunking
├── indexing.py              # Class for data embedding & indexing
├── retreival.py             # Class for RAG System: top k relevant chunks retrieval, augmentation & generation (invoking LLM)
├── app.py                   # Streamlit UI & Orchestration
├── config.py                # Classes for configuration settings (path, embedding model, LLM, vector DB, system prompt etc.)
├── logger_config.py         # Configuration settings for logger
├── main.py                  # To run the full ingestion & indexing pipeline
├── add_file.py              # To ingest & index a single file to an already existing vector DB
├── .env                     # API Keys (Private)
├── rag_system.log           # Logs for ingestion and RAG queries get saved here
└── requirements.txt         # Project dependencies
```

## ⚠️ Disclaimer
This project is for educational purposes. Ensure that any documents placed in the data/ folder do not contain sensitive personal information, as the contents are processed by the Google Gemini API (Free-tier data may be used by Google to train its API).


## :closed_lock_with_key: License

This project is intended for educational purposes only.

* **Usage:** You may review and learn from the code.
* **Restrictions:** Redistribution, commercial use, or modification for external projects require permission.
* **Ownership:** The work belongs to the project contributors and the university course.


## 🤝 Contributors
This project was developed as part of the Applied Computer Science Lab module in the BSc Computer Science degree programme at SRH Berlin University.


### Project Team:
* Nishant Malik
* Amrutha Manjunath
* Pruekson Lukkanothai
* Helin Men
* Nakai Mahachi

