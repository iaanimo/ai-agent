"""
Document Loader
================
Load and process documents from various formats:
  - Text files (.txt, .md)
  - PDF (.pdf)
  - Word (.docx)
  - Web pages (URL)
  - Directory (batch loading)
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



class DocumentLoader:
    """
    Multi-format document loader with automatic chunking.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def load_file(self, file_path: str) -> list[Document]:
        """Load a single file and split into chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in (".txt", ".md", ".py", ".js", ".json", ".csv", ".html"):
            return self._load_text(path)
        elif suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix == ".docx":
            return self._load_docx(path)
        else:
            # Try as text
            return self._load_text(path)

    def load_directory(self, directory: str, glob_pattern: str = "**/*.*") -> list[Document]:
        """Load all matching files from a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_docs = []
        for file_path in sorted(dir_path.glob(glob_pattern)):
            if file_path.is_file():
                try:
                    docs = self.load_file(str(file_path))
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return all_docs

    def load_text(self, text: str, metadata: dict = None) -> list[Document]:
        """Load raw text and split into chunks."""
        doc = Document(
            page_content=text,
            metadata=metadata or {"source": "direct_input"},
        )
        return self.splitter.split_documents([doc])

    def load_url(self, url: str) -> list[Document]:
        """Load content from a URL."""
        try:
            from bs4 import BeautifulSoup
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8")

            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            doc = Document(
                page_content=text,
                metadata={"source": url, "type": "web_page"},
            )
            return self.splitter.split_documents([doc])
        except Exception as e:
            raise RuntimeError(f"Failed to load URL: {e}")

    def _load_text(self, path: Path) -> list[Document]:
        """Load a text-based file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        doc = Document(
            page_content=content,
            metadata={"source": str(path), "filename": path.name, "type": path.suffix},
        )
        return self.splitter.split_documents([doc])

    def _load_pdf(self, path: Path) -> list[Document]:
        """Load a PDF file."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page": i + 1,
                            "type": "pdf",
                        },
                    )
                    docs.append(doc)
            return self.splitter.split_documents(docs)
        except ImportError:
            raise ImportError("PDF loading requires 'pypdf'. Install with: pip install pypdf")

    def _load_docx(self, path: Path) -> list[Document]:
        """Load a Word document."""
        try:
            import docx2txt
            text = docx2txt.process(str(path))
            doc = Document(
                page_content=text,
                metadata={"source": str(path), "filename": path.name, "type": "docx"},
            )
            return self.splitter.split_documents([doc])
        except ImportError:
            raise ImportError("DOCX loading requires 'docx2txt'. Install with: pip install docx2txt")
