
import os
import json
from src.basemodels import MD_Chunking_Model,PDF_Conversion_Model
from src.tools import convert_pdf_to_md_with_spinner, chunk_and_store


def pdf_models_from_file(path_to_file):
    with open(path_to_file, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
        for item in raw_data:
            yield PDF_Conversion_Model.model_validate(item)

def main():
    print("Starting up....")

    persist_directory = os.path.abspath(os.path.join(os.getcwd(), "gemma_chroma_db"))
    collection_name= "ollama_chroma_db"
    embedding_model= "embeddinggemma"
    embedding_dims=None
    ollama= True

    for pdf_model in pdf_models_from_file("./pdfs_to_convert.json"):
        print(f"Loading first PDF Model: {pdf_model.name}")
        # convert pdf to markdown
        md_mod = convert_pdf_to_md_with_spinner(pdf_model)
        print(f"{pdf_model.name} finished parsing.")

        print(f"{md_mod.name} starting to chunk.")
        chunk_and_store(md = md_mod,
                        persist_directory=persist_directory,
                        collection_name=collection_name,
                        embedding_dims=embedding_dims,
                        embedding_model=embedding_model,
                        ollama=ollama)
    


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()

    print(f"elapsed time {end-start}")
