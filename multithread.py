
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import json
from src.basemodels import MD_Chunking_Model,PDF_Conversion_Model
from src.tools import convert_pdf_to_md, chunk_and_store
import time
import argparse
from pathlib import Path

def markdown_check(pdf_model:PDF_Conversion_Model,markdown_dir:str):
    md_file = _build_md_path(pdf_model=pdf_model,markdown_dir=markdown_dir)
    if os.path.isfile(md_file):
        return True
    return False

def _build_md_path(pdf_model:PDF_Conversion_Model,markdown_dir:str):
    return os.path.join(markdown_dir,pdf_model.game_id,pdf_model.game_id+".md")

def _check_pdf_existence(pdf_model:str,pdf_dir:str)->bool:
    if os.path.isfile(os.path.join(pdf_dir,pdf_model.path)):
        return True
    return False

def main():
    print("Starting up....")

    parser = parser = argparse.ArgumentParser(description="Convers game PDFs to a Chroma DB for AI use.")
    parser.add_argument("source_file", help="Path or name of the source file containing json metadata for pdfs.")
    parser.add_argument("-p","--pdf_dir", required=True, help="Path to the pdf directory")
    parser.add_argument("-o","--output_dir", help="Path for markdown file destination.")
    parser.add_argument("-c","--chroma_dir", help="Path to the chroma directory")
    parser.add_argument("-d","--chroma_db", default="ollama_chroma_db",help="Database name.")
    parser.add_argument("-e","--embedding_model", default="embeddinggemma",help="Embedding model, eg embeddinggemme")


    # Parse the arguments from the terminal
    args = parser.parse_args()
    
    source_json = args.source_file if args.source_file else "./pdfs_to_convert.json"
    if not Path(source_json).is_file():
        parser.error(f"The source file '{source_json}' does not exist.")

    pdf_dir = args.pdf_dir
    if not Path(pdf_dir).is_dir():
        parser.error(f"The source folder '{pdf_dir}' does not exist.")

    md_output_dir = args.output_dir if args.output_dir else os.path.abspath(os.path.join(os.getcwd(),"markdown_files"))
    if not Path(md_output_dir):
        print(f"The output directory '{md_output_dir}' not found. Creating it now...")
        Path(md_output_dir).mkdir(parents=True,exist_ok=True)
    
    chroma_dir = args.chroma_dir if args.chroma_dir else os.path.abspath(os.path.join(os.getcwd(), "ollama_chroma_db"))
    if not Path(chroma_dir):
        print(f"The output directory '{chroma_dir}' not found. Creating it now...")
        Path(chroma_dir).mkdir(parents=True,exist_ok=True)


    collection_name= args.chroma_db if args.chroma_db else "ollama_chroma_db"
    embedding_model= args.embedding_model if args.embedding_model else "embeddinggemma"
    embedding_dims = None
    ollama = True
    pdf_models = []
    with open(source_json,"r",encoding="utf-8") as file:

        raw_data = json.load(file)

        for item in raw_data:
            pdf_models.append(PDF_Conversion_Model.model_validate(item))

    max_workers = 4
    cached_md_models = []
    futures = {}
    print('Starting Batch Multiprocessing PDF Conversion')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for pdf_model in pdf_models:

            if markdown_check(pdf_model, md_output_dir):
                print(f"ℹ️ [CACHE HIT] {pdf_model.name} markdown found. Skipping conversion.")
                
                # Instantiating the MD model programmatically using your existing PDF model data
                md_file_path = _build_md_path(pdf_model, md_output_dir)
                md_mod = MD_Chunking_Model(
                    **pdf_model.model_dump(exclude={"path"}),
                    path=md_file_path
                )
                cached_md_models.append(md_mod)
            elif _check_pdf_existence(pdf_model=pdf_model,pdf_dir = pdf_dir):
                # File doesn't exist, queue it up for parallel CPU processing
                print(f"⏳ [CACHE MISS] {pdf_model.name} needs conversion. Adding to queue.")
                future = executor.submit(convert_pdf_to_md, pdf_model, pdf_dir, md_output_dir)
                futures[future] = pdf_model
            else:
                print(f"❌ [ERROR] Source PDF for {pdf_model.name} is missing, and no cached markdown exists. Skipping.")
            

        if cached_md_models:
            print(f"\n--- Chunking {len(cached_md_models)} cached documents ---")
            for md_mod in cached_md_models:
                print(f"📦 {md_mod.name} starting to chunk (from cache).")
                chunk_and_store(md=md_mod,
                                persist_directory=chroma_dir,
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
                                persist_directory=chroma_dir,
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
