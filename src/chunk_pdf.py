import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/research.pdf"


def load_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number + 1,
                "text": text,
                "source": pdf_path
            })

    document.close()

    return pages


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"]
                }
            })

    return chunks


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)

    chunks = create_chunks(pages)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- CHUNK {i + 1} ---")
        print("Source:", chunk["metadata"]["source"])
        print("Page:", chunk["metadata"]["page"])
        print("Text:")
        print(chunk["text"][:500])