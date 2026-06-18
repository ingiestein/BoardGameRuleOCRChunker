from pathlib import Path

import os
from src.tools import chunk_and_store

from src.basemodels import MD_Chunking_Model
if __name__ == "__main__":
    import sys
    md_mod = MD_Chunking_Model(
            
        name ="Burning Banners",
        path = Path("Game PDFs/burning_banners/burning_banners.md"),
        game_id= "burning_banners",
        expansion_id =  None
    
    )
    print(md_mod.path)

    persist_directory = os.path.abspath(os.path.join(os.getcwd(), "gemma_chroma_db"))
    collection_name= "gemma_chroma_db"
    embedding_model= "embeddinggemma"
    embedding_dims=None
    ollama= True
    chunk_and_store(md = md_mod,
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_dims=embedding_dims,
                embedding_model=embedding_model,
                ollama=ollama)