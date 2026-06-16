from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class PDF_Conversion_Model(BaseModel):
    """Conversion Model"""
    game_id: str = Field(
        description="The name of the base board game. Leave null if unknown."
    )
    expansion_id: Optional[str] = Field(
        default=None,
        description="The name of the expansion. Leave null if unknown."
    )
    path: str = Field(
        description = "The path to the PDF"
    )
    name: str = Field(
        description = "The name of the game."
    )


class MD_Chunking_Model(BaseModel):
    """MD Chunking Model"""
    game_id: str = Field(
        description="The name of the base board game. Leave null if unknown."
    )
    expansion_id: Optional[str] = Field(
        default = None,
        description="The name of the expansion. Leave null if unknown."
    )
    path: str | Path = Field(
        description = "The path to the MD file"
    )
    name: str = Field(
        description = "The name of the game."
    )
    