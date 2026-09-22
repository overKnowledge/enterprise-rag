from dataclasses import dataclass


@dataclass
class PageContent:
    """One page (or page-equivalent unit) of extracted text."""

    page_number: int  # 1-indexed; use 1 for sources with no real pages (web)
    text: str


@dataclass
class RawDocument:
    """Output of any loader, regardless of source format."""

    source_type: str  # "pdf" | "docx" | "web"
    source_name: str  # original filename or URL
    content_hash: str  # sha256 of raw bytes — used as the document ID
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)