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


def critique_answer(
    question,
    answer,
    evidence
):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL
    )

    evidence_text = ""

    for i, chunk in enumerate(evidence, start=1):

        page = chunk.metadata.get("page", "Unknown")
        source = chunk.metadata.get("source", "Unknown")

        evidence_text += f"""
Evidence {i}

Source:
{source}

Page:
{page}

Text:
{chunk.page_content}

-----------------------------------
"""

    prompt = f"""
You are a Critic Agent in an
evidence-based research intelligence system.

Original Research Question:

{question}


Generated Answer:

{answer}


Retrieved Evidence:

{evidence_text}


Critically evaluate the generated answer.

Check:

1. Unsupported claims
2. Missing evidence
3. Citation accuracy
4. Contradictions
5. Completeness
6. Hallucination risk

Rules:

- Analyze only the supplied answer and evidence.
- Do not introduce outside knowledge.
- Do not invent problems.
- Distinguish genuine problems from missing information.

Return:

CRITIC ANALYSIS

OVERALL ASSESSMENT:
PASS / NEEDS REVISION

SUPPORTED CLAIMS:
- ...

UNSUPPORTED OR WEAK CLAIMS:
- ...

MISSING EVIDENCE:
- ...

CITATION ISSUES:
- ...

CONSISTENCY ISSUES:
- ...

HALLUCINATION RISK:
LOW / MEDIUM / HIGH

RECOMMENDATION:
...
"""

    response = llm.invoke(prompt)

    return extract_text(response)


if __name__ == "__main__":

    print("Critic Agent module created successfully.")