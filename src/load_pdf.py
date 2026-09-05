import fitz


PDF_PATH = "data/research.pdf"


def load_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)

    print(f"Total pages: {len(pages)}")

    for page in pages[:2]:
        print("\n--- PAGE", page["page"], "---")
        print(page["text"][:1000])