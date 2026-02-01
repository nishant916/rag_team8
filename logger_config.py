# logger_config.py
import logging
import sys
from config import settings

def setup_app_logging(mode: str = 'a'):
    """
    Centralized logging configuration.
    Parameters: 'w' for fresh overwrite (Ingestion), 'a' for append (Retrieval)
    """
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the function is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 1. File Handler (Writes to rag_system.log)
        file_handler = logging.FileHandler(settings.paths.LOG_FILE, mode=mode, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. Stream Handler (Writes to Terminal/Console)
        #stream_handler = logging.StreamHandler(sys.stdout)
        #stream_handler.setFormatter(formatter)
        #logger.addHandler(stream_handler)
    
    return logger

