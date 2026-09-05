import os
import time
import fitz

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

PDF_PATH = "data/research.pdf"
CHROMA_PATH = "chroma_db"


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


def create_vector_database(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_db = Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    batch_size = 20

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]

        metadatas = [
            chunk["metadata"]
            for chunk in batch
        ]

        print(
            f"Embedding chunks "
            f"{start + 1}-{start + len(batch)} "
            f"of {len(chunks)}..."
        )

        vector_db.add_texts(
            texts=texts,
            metadatas=metadatas
        )

        # Small delay between batches
        if start + batch_size < len(chunks):
            time.sleep(2)

    return vector_db


if __name__ == "__main__":

    print("Loading PDF...")

    pages = load_pdf(PDF_PATH)

    print(f"Loaded {len(pages)} pages")

    print("Creating chunks...")

    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings and storing in ChromaDB...")

    create_vector_database(chunks)

    print("\nChromaDB created successfully!")