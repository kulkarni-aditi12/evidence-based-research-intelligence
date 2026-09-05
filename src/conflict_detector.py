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


def detect_conflicts(question, retrieved_chunks):

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
You are a Conflict Detection Agent.

Research Question:
{question}

Retrieved Evidence:
{evidence_text}

Determine whether the evidence contains
meaningful relationships of:

- AGREEMENT
- CONFLICT
- UNCERTAIN

Rules:

- Analyze ONLY the provided evidence.
- Do not use outside knowledge.
- Do not invent conflicts.
- Similar information is not automatically a conflict.
- Different numerical values are not automatically a conflict.
- Consider whether differences could be caused by:
  datasets,
  populations,
  experiments,
  conditions,
  methods,
  or stages of research.
- Only classify something as CONFLICT when the
  evidence genuinely contradicts each other.

Return:

CONFLICT ANALYSIS

RELATIONSHIP 1:

Evidence A:
...

Page A:
...

Claim A:
...

Evidence B:
...

Page B:
...

Claim B:
...

Relationship:
AGREEMENT / CONFLICT / UNCERTAIN

Explanation:
...

If no meaningful conflict exists, write:

"No meaningful conflict detected."
"""

    response = llm.invoke(prompt)

    return extract_text(response)


if __name__ == "__main__":

    print("Conflict Detector module created successfully.")