import argparse
import sys
import re
import time
import threading
from pathlib import Path
import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    LayoutOptions, 
    RapidOcrOptions
)
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode

from src.basemodels import PDF_Conversion_Model, MD_Chunking_Model

def convert_pdf_to_md(pdf_md: PDF_Conversion_Model, pdf_dir:str, output_dir: str | None = None) -> Path:
    """
    Convert a PDF rulebook to Markdown format using Docling.
     - input_path: Path to the input PDF file.
     - output_dir: Optional directory to save the output. Defaults to the same directory as the PDF.
     - game_id: Optional identifier for the game. Defaults to the PDF filename without extension.
    """

    input_path = os.path.abspath(os.path.join(pdf_dir , pdf_md.path))
    game_id = pdf_md.game_id
    expansion_id = pdf_md.expansion_id
    name = pdf_md.name

    input_pdf = Path(input_path).resolve()

    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    
    #ensure it's a PDF file
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a PDF: {input_pdf}")
    
    if game_id is None:
        game_id = input_pdf.stem

    game_dir = re.sub(r"[^a-zA-Z0-9]+", "_", game_id.lower()).strip("_")

    if output_dir is not None:
        output_path = Path(output_dir).resolve() / game_dir
    else:
        output_path = input_pdf.parent / game_dir

    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / f"{game_dir}.md"
    artifacts_dir = output_path / f"{game_dir}_artifacts"
    
    print(f"Initializing Docling Engine with Heron Layout and Forced OCR...")
    
    # 1. Initialize Pipeline Options
    pipeline_options = PdfPipelineOptions()
    
    # 2. Generate all useful image artifacts for complex PDFs.
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = 2.0

    # Increase robustness for heavier/longer documents.
    pipeline_options.document_timeout = 600.0
    
    # 3. Explicitly set the new Heron Layout Model
    pipeline_options.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_HERON)
    
    # 4. Force Full-Page OCR using RapidOCR
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    
    # 5. Apply the configuration to the PDF Format Option
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    
    # Run the conversion (This will take significantly longer due to Forced OCR)
    result = converter.convert(str(input_pdf))
    

    # Save markdown with referenced images into artifacts_dir.
    # REFERENCED mode writes image files and links to them from markdown.
    result.document.save_as_markdown(
        markdown_path,
        artifacts_dir=artifacts_dir,
        image_mode=ImageRefMode.REFERENCED,
    )

    # Verify export quality: referenced image links should be present.
    md_text = markdown_path.read_text(encoding="utf-8")
    image_link_count = len(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_text))
    placeholder_count = md_text.count("<!-- image -->")

    print(
        f"ℹ️ Markdown image links: {image_link_count}, placeholders: {placeholder_count}"
    )
    if image_link_count == 0 and placeholder_count > 0:
        print(
            "⚠️ Export contains placeholders but no path references. "
            "Image links are not available for RAG display in this output."
        )
    

    print(f"✅ Conversion complete! Markdown: {markdown_path}")
    print(f"✅ Image artifacts saved to: {artifacts_dir}")
    
    md = MD_Chunking_Model(game_id = game_id,
                            expansion_id = expansion_id,
                            path = markdown_path,
                            name = name)
    
    
    return md
