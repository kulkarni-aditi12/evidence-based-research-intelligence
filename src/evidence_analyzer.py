from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

LLM_MODEL = "gemini-3.7-flash"


def extract_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])

            elif hasattr(item, "text"):
                text_parts.append(item.text)

        return "\n".join(text_parts)

    return str(content)


def analyze_evidence(question, retrieved_chunks):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL
    )

    evidence = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        page = chunk.metadata.get("page", "Unknown")
        source = chunk.metadata.get("source", "Unknown")

        evidence.append(
            f"""
Evidence {i}

Source:
{source}

Page:
{page}

Text:
{chunk.page_content}

-----------------------------------
"""
        )

    evidence_text = "\n".join(evidence)

    prompt = f"""
You are an Evidence Analysis Agent in a
research intelligence system.

Research Question:
{question}

Retrieved Evidence:
{evidence_text}

Analyze ONLY the retrieved evidence.

For every important claim that helps answer
the research question, identify:

1. Claim
2. Supporting Evidence
3. Page Number
4. Support Level
5. Evidence Strength

Support Level:

- Directly Supported
- Partially Supported
- Not Supported

Evidence Strength:

- Strong
- Moderate
- Weak

Rules:

- Use ONLY the supplied evidence.
- Do not use outside knowledge.
- Do not invent information.
- Do not invent page numbers.
- Do not make assumptions.
- Do not treat absence of information as proof.
- If evidence is insufficient, explicitly state so.

Return:

EVIDENCE ANALYSIS

CLAIM 1:

Claim:
...

Supporting Evidence:
...

Page:
...

Support:
...

Evidence Strength:
...

Continue for all important claims.

If the retrieved evidence is insufficient,
write:

"Insufficient evidence in the retrieved documents."
"""

    response = llm.invoke(prompt)

    return extract_text(response)


if __name__ == "__main__":

    print("Evidence Analyzer module created successfully.")