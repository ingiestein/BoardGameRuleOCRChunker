import re
import json
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document



from src.basemodels import MD_Chunking_Model


def _chunk_rulebooks(md: MD_Chunking_Model) -> Document:

    game_id = md.game_id
    expansion_id = md.expansion_id
    markdown_file_path = md.path
    name = md.name
    print(f"Chunking {name}")

    if not game_id or not game_id.strip():
        raise ValueError("game_id must be a non-empty string")
    if expansion_id is not None and not expansion_id.strip():
        raise ValueError("expansion_id must be a non-empty string when provided")

    
    md_path = Path(markdown_file_path).resolve()
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    
    
    headers_to_split_on = [("#","Chapter"), ("##","Section"), ("###","Subsection"), ("####","Subsubsection")]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)

    md_header_splits = markdown_splitter.split_text(markdown_content)


    #second pass: max character limits?

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " "],
    )

    final_chunks = text_splitter.split_documents(md_header_splits)

    image_link_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    game_title = game_id.replace("_", " ").title()
    expansion_title = expansion_id.replace("_", " ").title() if expansion_id else ""
    is_expansion = bool(expansion_id)

    for idx, chunk in enumerate(final_chunks):
        image_refs = image_link_pattern.findall(chunk.page_content)
        section = chunk.metadata.get("Section")
        chunk.metadata["game_id"] = game_id
        chunk.metadata["expansion_id"] = expansion_id or ""
        chunk.metadata["is_expansion"] = is_expansion
        chunk.metadata["ruleset_scope"] = "expansion" if is_expansion else "base"
        chunk.metadata["rule_priority"] = 2 if is_expansion else 1
        chunk.metadata["game_title"] = game_title
        chunk.metadata["expansion_title"] = expansion_title
        chunk.metadata["source_markdown"] = str(md_path)
        chunk.metadata["source_dir"] = str(md_path.parent)
        chunk.metadata["source_file"] = md_path.name
        chunk.metadata["section"] = section or "Unknown Section"
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["has_images"] = bool(image_refs)
        chunk.metadata["image_refs"] = json.dumps(image_refs)
    return final_chunks

def _store_chunks(
    chunks,
    persist_directory: Path,
    collection_name: str,
    embedding_model: str,
    embedding_dims: int | None,
    ollama: bool = False
                        ) -> None:
    
    print(f"Storing chunks in {collection_name}.\nModel: {embedding_model}.\nLocation: {persist_directory}\n")
    if not chunks:
        raise ValueError("No chunks were generated for ingestion")

    persist_directory.mkdir(parents=True, exist_ok=True)


    if ollama:
        if embedding_dims is not None:
            embeddings = OllamaEmbeddings(model=embedding_model, dimensions=embedding_dims)
        else:
            embeddings = OllamaEmbeddings(model=embedding_model)
    else:
        if embedding_dims is not None:
            embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dims)
        else:
            embeddings = OpenAIEmbeddings(model=embedding_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

def chunk_and_store(md: MD_Chunking_Model,
                    persist_directory: Path,
                    collection_name: str,
                    embedding_model: str,
                    embedding_dims: int | None,
                    ollama:bool = False)-> None:

    chunks = _chunk_rulebooks(md)
    _store_chunks(chunks=chunks, 
                    collection_name=collection_name,
                    persist_directory=Path(persist_directory),
                    embedding_dims=embedding_dims,
                    embedding_model=embedding_model,
                    ollama=ollama)



