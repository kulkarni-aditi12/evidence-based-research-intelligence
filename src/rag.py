from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from research_planner import create_research_plan
from evidence_analyzer import analyze_evidence
from conflict_detector import detect_conflicts
from critic_agent import critique_answer
from synthesis_agent import generate_final_answer

from research_memory import (
    save_memory,
    search_memory
)


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "gemini-3.7-flash"

EMBEDDING_MODEL = "gemini-embedding-001"

CHROMA_PATH = "chroma_db"


# ============================================================
# STATE
# ============================================================

class ResearchState(TypedDict):

    question: str

    research_plan: str

    retrieved_chunks: List

    evidence_analysis: str

    conflict_analysis: str

    draft_answer: str

    critic_analysis: str

    final_answer: str

    memory_results: List


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL
)


# ============================================================
# VECTOR DATABASE
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)


vector_db = Chroma(
    collection_name="research_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


# ============================================================
# NODE 1 — RESEARCH PLANNER
# ============================================================

def planner_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 1 — RESEARCH PLANNER")
    print("=" * 70)

    question = state["question"]

    plan = create_research_plan(
        question
    )

    print("\nResearch Plan:")
    print(plan)

    return {
        "research_plan": plan
    }


# ============================================================
# NODE 2 — EVIDENCE RETRIEVAL
# ============================================================

def retrieval_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 2 — EVIDENCE RETRIEVAL")
    print("=" * 70)

    question = state["question"]

    research_plan = state[
        "research_plan"
    ]

    # Extract individual questions
    subquestions = []

    for line in research_plan.splitlines():

        line = line.strip()

        if not line:
            continue

        if line[0].isdigit():

            question_text = line

            if "." in question_text:

                question_text = question_text.split(
                    ".",
                    1
                )[1].strip()

            subquestions.append(
                question_text
            )

    retrieved_chunks = []

    # Retrieve for original question
    original_results = vector_db.similarity_search(
        question,
        k=3
    )

    retrieved_chunks.extend(
        original_results
    )

    # Retrieve for each subquestion
    for subquestion in subquestions:

        print(
            f"\nRetrieving evidence for:\n{subquestion}"
        )

        results = vector_db.similarity_search(
            subquestion,
            k=3
        )

        retrieved_chunks.extend(
            results
        )

    # Remove duplicate chunks
    unique_chunks = []

    seen = set()

    for chunk in retrieved_chunks:

        page = chunk.metadata.get(
            "page",
            "Unknown"
        )

        source = chunk.metadata.get(
            "source",
            "Unknown"
        )

        text = chunk.page_content

        key = (
            source,
            page,
            text
        )

        if key not in seen:

            seen.add(key)

            unique_chunks.append(
                chunk
            )

    print(
        f"\nTotal unique evidence chunks: "
        f"{len(unique_chunks)}"
    )

    return {
        "retrieved_chunks":
            unique_chunks
    }


# ============================================================
# NODE 3 — EVIDENCE ANALYZER
# ============================================================

def evidence_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 3 — EVIDENCE ANALYZER")
    print("=" * 70)

    analysis = analyze_evidence(
        state["question"],
        state["retrieved_chunks"]
    )

    print("\nEvidence Analysis:")
    print(analysis)

    return {
        "evidence_analysis":
            analysis
    }


# ============================================================
# NODE 4 — CONFLICT DETECTOR
# ============================================================

def conflict_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 4 — CONFLICT DETECTOR")
    print("=" * 70)

    conflicts = detect_conflicts(
        state["question"],
        state["retrieved_chunks"]
    )

    print("\nConflict Analysis:")
    print(conflicts)

    return {
        "conflict_analysis":
            conflicts
    }


# ============================================================
# NODE 5 — DRAFT GENERATION
# ============================================================

def draft_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 5 — DRAFT GENERATOR")
    print("=" * 70)

    prompt = f"""
You are a Research Drafting Agent.

Research Question:

{state["question"]}

Research Plan:

{state["research_plan"]}

Evidence Analysis:

{state["evidence_analysis"]}

Conflict Analysis:

{state["conflict_analysis"]}

Create a preliminary research answer.

Rules:

- Use ONLY the supplied information.
- Do not use outside knowledge.
- Do not invent citations.
- Do not invent page numbers.
- Prefer evidence-supported claims.
- Clearly distinguish uncertain information.
- Do not hide conflicts.

Create a concise academic draft.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):

        draft = content

    else:

        draft = str(content)

    print("\nDraft Answer:")
    print(draft)

    return {
        "draft_answer":
            draft
    }


# ============================================================
# NODE 6 — CRITIC AGENT
# ============================================================

def critic_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 6 — CRITIC AGENT")
    print("=" * 70)

    critique = critique_answer(
        state["question"],
        state["draft_answer"],
        state["retrieved_chunks"]
    )

    print("\nCritic Analysis:")
    print(critique)

    return {
        "critic_analysis":
            critique
    }


# ============================================================
# NODE 7 — FINAL SYNTHESIS
# ============================================================

def synthesis_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 7 — FINAL SYNTHESIS AGENT")
    print("=" * 70)

    final_answer = generate_final_answer(

        question=
            state["question"],

        research_plan=
            state["research_plan"],

        evidence_analysis=
            state["evidence_analysis"],

        conflict_analysis=
            state["conflict_analysis"],

        draft_answer=
            state["draft_answer"],

        critic_analysis=
            state["critic_analysis"]
    )

    return {
        "final_answer":
            final_answer
    }


# ============================================================
# NODE 8 — RESEARCH MEMORY
# ============================================================

def memory_node(state: ResearchState):

    print("\n")
    print("=" * 70)
    print("AGENT 8 — RESEARCH MEMORY")
    print("=" * 70)

    save_memory(

        question=
            state["question"],

        research_plan=
            state["research_plan"],

        final_answer=
            state["final_answer"]
    )

    print(
        "\nResearch successfully stored in memory."
    )

    return {
        "memory_results":
            search_memory(
                state["question"]
            )
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        ResearchState
    )

    # Add nodes

    graph.add_node(
        "planner",
        planner_node
    )

    graph.add_node(
        "retrieval",
        retrieval_node
    )

    graph.add_node(
        "evidence",
        evidence_node
    )

    graph.add_node(
        "conflict",
        conflict_node
    )

    graph.add_node(
        "draft",
        draft_node
    )

    graph.add_node(
        "critic",
        critic_node
    )

    graph.add_node(
        "synthesis",
        synthesis_node
    )

    graph.add_node(
        "memory",
        memory_node
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "planner"
    )

    graph.add_edge(
        "planner",
        "retrieval"
    )

    graph.add_edge(
        "retrieval",
        "evidence"
    )

    graph.add_edge(
        "evidence",
        "conflict"
    )

    graph.add_edge(
        "conflict",
        "draft"
    )

    graph.add_edge(
        "draft",
        "critic"
    )

    graph.add_edge(
        "critic",
        "synthesis"
    )

    graph.add_edge(
        "synthesis",
        "memory"
    )

    graph.add_edge(
        "memory",
        END
    )

    return graph.compile()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("      EVIDENCE-BASED RESEARCH INTELLIGENCE SYSTEM")
    print("      LANGGRAPH MULTI-AGENT PIPELINE")
    print("#" * 70)

    question = input(
        "\nEnter your research question: "
    )

    if not question.strip():

        print(
            "\nResearch question cannot be empty."
        )

        return

    # Build graph

    app = build_graph()

    # Initial state

    initial_state = {

        "question":
            question,

        "research_plan":
            "",

        "retrieved_chunks":
            [],

        "evidence_analysis":
            "",

        "conflict_analysis":
            "",

        "draft_answer":
            "",

        "critic_analysis":
            "",

        "final_answer":
            "",

        "memory_results":
            []
    }

    # Run graph

    result = app.invoke(
        initial_state
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n\n")
    print("#" * 70)
    print("                    FINAL REPORT")
    print("#" * 70)

    print(
        result["final_answer"]
    )

    print("\n")
    print("#" * 70)
    print("                    PIPELINE COMPLETE")
    print("#" * 70)


if __name__ == "__main__":

    main()