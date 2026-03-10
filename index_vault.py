import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ===== CHANGE THIS PATH =====
VAULT_PATH = r"C:\Users\ibane\OneDrive\Documentos\Obsidian Vault"
PERSIST_DIR = "data"

# Skip folders you don't want indexed
SKIP_DIRS = {
    ".obsidian",
    ".git",
    "node_modules",
}

# Only index these file extensions
ALLOWED_EXTS = {".md"}


def is_in_skipped_dir(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(sd.lower() in parts for sd in SKIP_DIRS)


def read_text_robust(file_path: Path) -> str | None:
    """
    Try to read a markdown file with multiple encodings.
    Return None if it cannot be read.
    """
    # Common encodings for Obsidian/Windows
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    for enc in encodings_to_try:
        try:
            return file_path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue

    # Last resort: read with replacement so we don't crash
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def load_documents() -> list[Document]:
    vault = Path(VAULT_PATH)

    docs: list[Document] = []
    skipped: list[str] = []

    for fp in vault.rglob("*"):
        if fp.is_dir():
            continue
        if fp.suffix.lower() not in ALLOWED_EXTS:
            continue
        if is_in_skipped_dir(fp):
            continue

        text = read_text_robust(fp)
        if not text or not text.strip():
            skipped.append(str(fp))
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(fp),
                    "filename": fp.name,
                },
            )
        )

    print(f"Loaded {len(docs)} markdown files.")
    if skipped:
        print(f"Skipped {len(skipped)} files (unreadable/empty). Showing up to 5:")
        for s in skipped[:5]:
            print("  -", s)

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    final_chunks: list[Document] = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        filename = doc.metadata.get("filename", "Unknown file")
        content = doc.page_content

        try:
            md_sections = markdown_splitter.split_text(content)
        except Exception:
            md_sections = [Document(page_content=content, metadata={})]

        for section in md_sections:
            section_text = section.page_content.strip()
            if not section_text:
                continue

            section_metadata = {
                "source": source,
                "filename": filename,
            }

            section_metadata.update(section.metadata)

            smaller_chunks = recursive_splitter.create_documents(
                texts=[section_text],
                metadatas=[section_metadata]
            )

            final_chunks.extend(smaller_chunks)

    for i, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = i + 1

    return final_chunks


def create_index(chunks: list[Document]) -> None:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    db.persist()


def main():
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and storing in Chroma...")
    create_index(chunks)

    print("Done. Index stored in ./data")


if __name__ == "__main__":
    main()