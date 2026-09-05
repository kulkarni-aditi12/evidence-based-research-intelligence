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


def create_research_plan(question):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL
    )

    prompt = f"""
You are a Research Planner Agent.

Your job is to break a research question into
specific subquestions that can be answered using
research documents.

Research Question:
{question}

Generate 4 focused research subquestions.

The questions should help investigate:

1. The main problem or research gap
2. The objectives or motivation
3. The methodology or proposed approach
4. The results, applications, implications, or limitations

Rules:

- Keep the questions directly related to the research question.
- Do not answer the questions.
- Do not introduce outside knowledge.
- Return only the research questions.

Format:

1. ...
2. ...
3. ...
4. ...
"""

    response = llm.invoke(prompt)

    return extract_text(response)


if __name__ == "__main__":

    question = input("Enter research question: ")

    plan = create_research_plan(question)

    print("\nRESEARCH PLAN")
    print("=" * 70)
    print(plan)