
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import json
from src.basemodels import MD_Chunking_Model,PDF_Conversion_Model
from src.tools import convert_pdf_to_md_with_spinner, chunk_and_store
import time

def markdown_check(pdf_model:PDF_Conversion_Model,markdown_dir:str):
    md_file = _build_md_path(pdf_model=pdf_model,markdown_dir=markdown_dir)
    if os.path.isfile(md_file):
        return True
    return False

def _build_md_path(pdf_model:PDF_Conversion_Model,markdown_dir:str):
    return os.path.join(markdown_dir,pdf_model.game_id,pdf_model.game_id+".md")

def _check_pdf_existence(pdf_model):
    if os.path.isfile(pdf_model.path):
        return True
    return False

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
    cached_md_models = []
    futures = {}
    print('Starting Batch Multiprocessing PDF Conversion')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for pdf_model in pdf_models:

            if markdown_check(pdf_model, md_dir):
                print(f"ℹ️ [CACHE HIT] {pdf_model.name} markdown found. Skipping conversion.")
                
                # Instantiating the MD model programmatically using your existing PDF model data
                md_file_path = _build_md_path(pdf_model, md_dir)
                md_mod = MD_Chunking_Model(
                    **pdf_model.model_dump(exclude={"path"}),
                    path=md_file_path
                )
                cached_md_models.append(md_mod)
            elif _check_pdf_existence(pdf_model=pdf_model):
                # File doesn't exist, queue it up for parallel CPU processing
                print(f"⏳ [CACHE MISS] {pdf_model.name} needs conversion. Adding to queue.")
                future = executor.submit(convert_pdf_to_md_with_spinner, pdf_model, md_dir)
                futures[future] = pdf_model
            else:
                print(f"❌ [ERROR] Source PDF for {pdf_model.name} is missing, and no cached markdown exists. Skipping.")
            

        if cached_md_models:
            print(f"\n--- Chunking {len(cached_md_models)} cached documents ---")
            for md_mod in cached_md_models:
                print(f"📦 {md_mod.name} starting to chunk (from cache).")
                chunk_and_store(md=md_mod,
                                persist_directory=persist_directory,
                                collection_name=collection_name,
                                embedding_dims=embedding_dims,
                                embedding_model=embedding_model,
                                ollama=ollama)

        if futures:
            print("\n--- Waiting for active PDF conversions to complete ---")
            for future in as_completed(futures):
                original_pdf_model = futures[future]
                try:
                    md_mod = future.result() 
                    print(f"{md_mod.name} finished parsing.")
                    print(f"{md_mod.name} starting to chunk.")

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
