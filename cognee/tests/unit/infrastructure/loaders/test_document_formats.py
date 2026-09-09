"""Exercise default native document conversion with the runtime's Docling extra."""

from pathlib import Path
import pytest
from cognee.infrastructure.loaders import get_loader_engine

DATA = Path(__file__).parents[3] / "test_data"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,loader_name,expected",
    [
        ("artificial-intelligence.pdf", "pypdf_loader", "Page 1:"),
        ("example_with_header.csv", "csv_loader", "Row 1:"),
        ("example.docx", "docling_loader", "Paragraph"),
        ("example.pptx", "docling_loader", "Column 1"),
        ("example.xlsx", "docling_loader", None),
    ],
)
async def test_native_document_loader_returns_content(filename, loader_name, expected):
    path = DATA / filename
    loader = get_loader_engine().get_loader(str(path), preferred_loaders=None)
    assert loader is not None
    assert loader.loader_name == loader_name
    content = await loader.load(str(path), persist=False)
    assert isinstance(content, str) and content.strip()
    if expected:
        assert expected in content
