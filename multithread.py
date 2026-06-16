
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import json
from src.basemodels import MD_Chunking_Model,PDF_Conversion_Model
from src.tools import convert_pdf_to_md_with_spinner, chunk_and_store
import time

def main():
    print("Starting up....")

    persist_directory = os.path.abspath(os.path.join(os.getcwd(), "gemma_chroma_db"))
    md_dir = os.path.abspath(os.path.join(os.getcwd(),"markdown_files"))
    collection_name= "ollama_chroma_db"
    embedding_model= "embeddinggemma"
    embedding_dims=None
    ollama= True
    pdf_models = []
    with open("./pdfs_to_convert.json","r",encoding="utf-8") as file:

        raw_data = json.load(file)

        for item in raw_data:
            pdf_models.append(PDF_Conversion_Model.model_validate(item))

    max_workers = 4
    print('Starting Batch Multiprocessing PDF Conversion')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_pdf_to_md_with_spinner, pdf_model, md_dir): pdf_model for pdf_model in pdf_models}
        for future in as_completed(futures):
            original_pdf_model = futures[future]
            try:
                
                md_mod = future.result() 
                
                print(f"{md_mod.name} finished parsing.")
                print(f"{md_mod.name} starting to chunk.")
                
                # Note: This step is running sequentially in your main process
                chunk_and_store(md = md_mod,
                            persist_directory=persist_directory,
                            collection_name=collection_name,
                            embedding_dims=embedding_dims,
                            embedding_model=embedding_model,
                            ollama=ollama)
                            
            except Exception as exc:
                print(f"{original_pdf_model.name} generated an exception: {exc}")


if __name__ == "__main__":

    start = time.time()
    main()
    end = time.time()

    print(f"elapsed time {end-start}")
