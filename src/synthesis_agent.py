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


def generate_final_answer(
    question,
    research_plan,
    evidence_analysis,
    conflict_analysis,
    draft_answer,
    critic_analysis
):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL
    )

    prompt = f"""
You are the Final Synthesis Agent.

Your task is to produce the final
evidence-based research report.

==================================================
RESEARCH QUESTION
==================================================

{question}

==================================================
RESEARCH PLAN
==================================================

{research_plan}

==================================================
EVIDENCE ANALYSIS
==================================================

{evidence_analysis}

==================================================
CONFLICT ANALYSIS
==================================================

{conflict_analysis}

==================================================
DRAFT ANSWER
==================================================

{draft_answer}

==================================================
CRITIC ANALYSIS
==================================================

{critic_analysis}

==================================================
RULES
==================================================

1. Use ONLY the supplied information.

2. Do not introduce outside knowledge.

3. Correct unsupported claims identified by
   the critic.

4. Remove claims that are not supported.

5. Preserve well-supported claims.

6. Do not invent citations.

7. Do not invent page numbers.

8. Mention page numbers for important claims.

9. Clearly identify conflicts when they exist.

10. Do not exaggerate conflicts.

11. If evidence is insufficient, explicitly say so.

12. Keep the report academically structured.

==================================================
FINAL FORMAT
==================================================

FINAL RESEARCH REPORT

1. RESEARCH QUESTION

State the research question.

2. RESEARCH METHODOLOGY

Explain how the evidence was retrieved
and analyzed.

3. KEY FINDINGS

Present the major evidence-supported findings.

For important findings include:

Claim:
...

Evidence:
...

Page:
...

Evidence Strength:
...

4. EVIDENCE ASSESSMENT

Explain how strongly the evidence supports
the main findings.

5. AGREEMENT AND CONFLICTS

Discuss meaningful agreements or conflicts.

If none exist:

"No meaningful conflict detected."

6. LIMITATIONS

Mention only limitations supported by
the evidence.

If none are explicitly identified:

"No explicit limitations identified in
the retrieved evidence."

7. CONCLUSION

Directly answer the research question using
the strongest available evidence.

8. SOURCES

List the pages and source documents used.

Format:

- Page X | source
"""

    response = llm.invoke(prompt)

    return extract_text(response)


if __name__ == "__main__":

    print("Synthesis Agent module created successfully.")