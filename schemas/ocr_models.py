import enum

from pydantic import BaseModel, Field, conlist, model_validator


class LanguageTag(enum.StrEnum):
    """Enumeration for the identified language column."""

    GEEZ = "Geez"
    AMHARIC = "Amharic"
    ENGLISH = "English"
    UNKNOWN = "Unknown"


class BoundingBox(BaseModel):
    """Coordinates representing a rectangular area on a document page.

    Format: [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right.
    """

    coordinates: conlist(item_type=float, min_length=4, max_length=4) = Field(
        ..., description="[x1, y1, x2, y2] coordinates"
    )

    @property
    def x1(self) -> float:
        """Top-left X coordinate."""
        return self.coordinates[0]

    @property
    def y1(self) -> float:
        """Top-left Y coordinate."""
        return self.coordinates[1]

    @property
    def x2(self) -> float:
        """Bottom-right X coordinate."""
        return self.coordinates[2]

    @property
    def y2(self) -> float:
        """Bottom-right Y coordinate."""
        return self.coordinates[3]

    @property
    def center_x(self) -> float:
        """Calculate the horizontal center of the bounding box."""
        return (self.x1 + self.x2) / 2


class TextLine(BaseModel):
    """A single logical line of extracted text and its associated location."""

    text: str = Field(..., description="The raw OCR extracted text")
    bbox: BoundingBox = Field(..., description="The geometric bounds of this line")
    confidence: float | None = Field(None, description="OCR engine confidence score (0.0 to 1.0)")


class ColumnBlock(BaseModel):
    """A logical group of TextLines forming a column.

    The language is automatically detected based on the column's geometric position
    relative to the total page width.
    """

    lines: list[TextLine] = Field(default_factory=list, description="Text lines within this column")
    bbox: BoundingBox = Field(..., description="The geometric bounds encompassing all lines")
    language: LanguageTag = Field(
        default=LanguageTag.UNKNOWN, description="The detected language based on layout"
    )
    page_width: float = Field(
        ..., exclude=True, description="The width of the source image for relative calculation"
    )

    @model_validator(mode="after")
    def detect_language_geometrically(self) -> "ColumnBlock":
        """Geographic Language Clustering based on coordinates.

        Assumes a triple-column layout:
        - X < 33% -> Ge'ez
        - 33% <= X < 66% -> Amharic
        - X >= 66% -> English
        """
        if not self.bbox or not self.page_width:
            return self

        center_x = self.bbox.center_x
        relative_x = center_x / self.page_width

        if relative_x < 0.333:
            self.language = LanguageTag.GEEZ
        elif relative_x < 0.666:
            self.language = LanguageTag.AMHARIC
        else:
            self.language = LanguageTag.ENGLISH

        return self


class PageLayout(BaseModel):
    """The master container representing the successfully parsed structure of a single page."""

    page_number: int = Field(..., description="The physical page number from the PDF")
    image_width: float = Field(..., description="Width of the source image in pixels")
    image_height: float = Field(..., description="Height of the source image in pixels")
    columns: list[ColumnBlock] = Field(
        default_factory=list, description="The discrete language columns on this page"
    )
