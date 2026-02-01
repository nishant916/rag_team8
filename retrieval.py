import logging
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import settings
from logger_config import setup_app_logging

# Setup logging
setup_app_logging(mode='a') 
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        # 1. Load the same embedding model used in indexing.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.models.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. Connect to the existing ChromaDB index
        self.vectorstore = Chroma(
            persist_directory=settings.paths.VECTOR_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=settings.vector_db.COLLECTION_NAME
        )
        
        # 3. Initialize Gemini-3-Flash for Answer Generation
        self.llm = ChatGoogleGenerativeAI(
            model=settings.models.LLM_MODEL,
            google_api_key=settings.google.GOOGLE_API_KEY,
            temperature=settings.models.LLM_TEMPERATURE,
            max_output_tokens=settings.models.LLM_MAX_OUTPUT_TOKENS
        )

    def get_answer(self, query: str) -> str:
        """
        Executes the full RAG Chain: Retrieve Relevant Chunks -> Augment Prompt -> Generate Answer.
        Also logs the query, retreived chunks and similarity scores.
        """
        logger.info(f"Query: {query}")

        # Define the prompt structure
        prompt = ChatPromptTemplate.from_template("""
        {system_prompt}
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        USER QUESTION: {question}
        
        YOUR ANSWER:""")
        
        # Define the retriever to return scores
        retriever_with_scores = RunnableLambda(
            lambda q: self.vectorstore.similarity_search_with_relevance_scores(
                q, k=settings.vector_db.K_RETRIEVAL
            )
        )

        # Build the RAG Chain using LCEL (LangChain Expression Language)
        # 1: dictionary {retrieved docs with scores -> formatted text; question; system_prompt} ->
        # 2: prompt template -> 
        # 3: llm -> 
        # 4: output text
        rag_chain = (
            {"context": retriever_with_scores | self._format_docs, 
            "question": RunnablePassthrough(),
            "system_prompt": lambda x: settings.models.LLM_SYSTEM_PROMPT}      # Created dictionary with 3 keys: context, question, system_prompt
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            return rag_chain.invoke(query)
        except Exception as e:
            logger.error(f"RAG Chain execution failed: {e}")
            return "I'm sorry, I encountered an error while searching the documents."

    def _format_docs(self, docs_with_scores: List[Any]) -> str:
        """Helper to combine chunks and append their source metadata."""
        logger.info(f"\n--- Retrieved Top {len(docs_with_scores)} Chunks ---")
        formatted_context = []
        
        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get("source", "Unknown Source")
            
            # Clean the snippet to avoid newlines in the log
            clean_snippet = doc.page_content[:250].replace('\n', ' ')
            
            # Now the f-string only contains the clean variable
            logger.info(
                f"Chunk {i+1} | Score: {score:.4f} | Source: {source} | "
                f"\nSnippet: {clean_snippet}..."
            )
            
            formatted_context.append(f"Content: {doc.page_content}\n(Source: {source})\n")
            
        return "\n---\n".join(formatted_context)
