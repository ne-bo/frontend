HASHES_FILE = "backend/processed_hashes.json"
FAISS_PATH = "backend/faiss_index"
PROMPT_TEMPLATE = """You are the best documentation reader and explanator.
    Use the context to answer the research query from the user. 
    Use markdown in your answer. For example bullet points or titles.
    If you do not have enough information, answer that you need more documentation. 
    
    Context: {context}
    
    Research query: {input}
    """
OPENAI_MODEL_NAME = "gpt-3.5-turbo-0125"
TEMPERATURE = 0
CHUNKS_TO_RETRIVE = 3
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
