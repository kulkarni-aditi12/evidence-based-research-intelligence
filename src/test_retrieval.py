from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"


def test_retrieval():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_db = Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    question = input("\nEnter your research question: ")

    results = vector_db.similarity_search(question, k=3)

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        print(f"\n--- RESULT {i} ---")
        print(f"Page: {result.metadata.get('page')}")
        print(f"Source: {result.metadata.get('source')}")
        print("\nText:")
        print(result.page_content)
        print("-" * 70)


if __name__ == "__main__":
    test_retrieval()